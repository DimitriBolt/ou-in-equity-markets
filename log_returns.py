import glob
from pathlib import Path

import numpy as np
import pandas as pd
import websockets  # ! Нужен для yfinance в некоторых окружениях
import yfinance as yf
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


class LogReturns:
    # Список тикеров, которые мы заранее исключаем вручную.
    # Это именно учебный, ручной фильтр: он простой, понятный и легко контролируется исследователем.
    DEFAULT_EXCLUDED_SYMBOLS: list[str] = (
        ['STRD', 'STRF', 'STRC', 'BRKRP', 'STRK', 'SNDK', 'CRCL', 'MCHPP', 'AGNCZ', 'SAIL', 'GLXY', 'CRWV']
        + ['BRK/A', 'BRK/B']
        + ['EMP', 'FI', 'MMC']
        + ['ALAB', 'ASTS', 'AUR', 'BE', 'CRDO', 'FTAI', 'HIMS', 'IONQ', 'MP', 'OKLO', 'RGTI', 'RKLB',
           'SATS', 'SMCI', 'SMR', 'TEM', 'TTD']
        + ['AFRM', 'APP', 'CNC', 'COHR', 'COIN', 'CVNA', 'HOOD', 'JOBY', 'LITE', 'MDB', 'MRVL', 'MSTR',
           'PLTR', 'RDDT', 'SYM', 'U', 'VRT', 'VST', 'W']
        + ['CCZ']
    )

    def __init__(
        self,
        symbols_local: list[str],
        start_date_local: str,
        end_date_local: str,
        excluded_symbols_local: list[str] | None = None,
    ) -> None:
        # Сохраняем входные параметры объекта.
        self.symbols: list[str] = symbols_local
        self.start_date: str = start_date_local
        self.end_date: str = end_date_local

        # Если пользователь не передал свой список исключений,
        # используем простой учебный список по умолчанию.
        self.excluded_symbols: list[str] = (
            excluded_symbols_local if excluded_symbols_local is not None else self.DEFAULT_EXCLUDED_SYMBOLS
        )

        # Ниже — кэш вычисленных таблиц.
        # Идея lazy-подхода такая:
        # таблицы не считаются в __init__, а строятся только тогда, когда реально понадобились.
        self._prices_df: DataFrame | None = None
        self._log_prices_df: DataFrame | None = None
        self._log_returns_df: DataFrame | None = None
        self._detrended_log_prices_df: DataFrame | None = None
        self._detrended_log_returns_df: DataFrame | None = None

        # Здесь сохраним список тикеров, которые реально пойдут в download.
        # Это полезно и для контроля, и для отладки.
        self._filtered_symbols: list[str] | None = None
        self._dropped_symbols_with_missing_history: list[str] = []
        self._dropped_symbols_with_non_positive_prices: list[str] = []

    def _get_filtered_symbols(self) -> list[str]:
        # Удаляем вручную исключённые тикеры.
        # Это предельно простая и прозрачная логика без лишней автоматизации.
        if self._filtered_symbols is None:
            self._filtered_symbols = [symbol for symbol in self.symbols if symbol not in self.excluded_symbols]
        return self._filtered_symbols

    def _download_prices(self) -> DataFrame:
        # Если таблица цен уже была скачана раньше, просто возвращаем кэш.
        if self._prices_df is not None:
            return self._prices_df

        filtered_symbols: list[str] = self._get_filtered_symbols()

        if len(filtered_symbols) == 0:
            raise ValueError("После исключения проблемных тикеров список symbols оказался пустым.")

        # Скачиваем цены закрытия с автоматической поправкой.
        downloaded = yf.download(
            tickers=filtered_symbols,
            start=self.start_date,
            end=self.end_date,
            timeout=20,
            auto_adjust=True,
            progress=True,
        )

        # При нескольких тикерах yfinance часто возвращает MultiIndex-таблицу,
        # и тогда нужный нам слой — это 'Close'.
        # При одном тикере структура может быть другой, поэтому ниже делаем
        # простую и понятную нормализацию формата.
        if isinstance(downloaded.columns, pd.MultiIndex):
            prices = downloaded['Close']
        else:
            # Если тикер один, yfinance может вернуть обычную таблицу с колонкой Close.
            if 'Close' in downloaded.columns:
                prices = downloaded[['Close']].copy()
                prices.columns = filtered_symbols[:1]
            else:
                # Запасной вариант: если формат неожиданно отличается,
                # всё равно превращаем результат в DataFrame.
                prices = pd.DataFrame(downloaded)

        # Если после скачивания пришёл Series, превращаем его в DataFrame.
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=filtered_symbols[0])

        # На всякий случай приводим названия колонок к строкам.
        prices.columns = [str(column) for column in prices.columns]

        # Удаляем строки, где вообще всё пусто.
        # Это простая и безопасная операция, не усложняющая код.
        prices = prices.dropna(how='all')

        # Для дальнейшего анализа нужен полный положительный ценовой panel:
        # пропуски ломают detrending, а неположительные цены несовместимы с log.
        missing_history_mask = prices.isna().any(axis=0)
        self._dropped_symbols_with_missing_history = prices.columns[missing_history_mask].tolist()
        prices = prices.loc[:, ~missing_history_mask]

        non_positive_mask = (prices <= 0).any(axis=0)
        self._dropped_symbols_with_non_positive_prices = prices.columns[non_positive_mask].tolist()
        prices = prices.loc[:, ~non_positive_mask]

        if prices.empty:
            raise ValueError("Не удалось получить данные цен: таблица prices пустая.")

        self._prices_df = prices
        return self._prices_df

    def _build_log_prices(self) -> DataFrame:
        # Если лог-цены уже построены, возвращаем кэш.
        if self._log_prices_df is not None:
            return self._log_prices_df

        prices_df: DataFrame = self.get_prices()

        # Берём натуральный логарифм цен.
        # Используем apply(np.log), чтобы результат явно оставался DataFrame.
        self._log_prices_df = prices_df.apply(np.log)

        return self._log_prices_df

    def _build_log_returns(self) -> DataFrame:
        # Если обычные log returns уже рассчитаны, возвращаем их из кэша.
        if self._log_returns_df is not None:
            return self._log_returns_df

        log_prices_df: DataFrame = self.get_log_prices()

        # Это и есть обычные log returns:
        # log(S_t) - log(S_{t-1})
        self._log_returns_df = log_prices_df.diff().dropna()

        return self._log_returns_df

    def _build_detrended_log_prices(self) -> DataFrame:
        # Если detrended log-prices уже есть, не пересчитываем их.
        if self._detrended_log_prices_df is not None:
            return self._detrended_log_prices_df

        log_prices_df: DataFrame = self.get_log_prices()

        # Создаём временной индекс: 0, 1, 2, ..., T-1.
        # Именно его используем как независимую переменную для линейного тренда.
        x_time = np.arange(len(log_prices_df)).reshape(-1, 1)

        # Здесь каждая колонка log-prices рассматривается как зависимая переменная.
        # sklearn умеет в таком случае оценить линейный тренд сразу по всем колонкам.
        model = LinearRegression()
        model.fit(x_time, log_prices_df)

        # Предсказываем линейный тренд и вычитаем его из log-prices.
        trend = model.predict(x_time)
        trend_df: DataFrame = pd.DataFrame(
            trend,
            index=log_prices_df.index,
            columns=log_prices_df.columns,
        )

        self._detrended_log_prices_df = log_prices_df - trend_df

        return self._detrended_log_prices_df

    def _build_detrended_log_returns(self) -> DataFrame:
        # Если detrended log returns уже есть, возвращаем кэш.
        if self._detrended_log_returns_df is not None:
            return self._detrended_log_returns_df

        detrended_log_prices_df: DataFrame = self.get_detrended_log_prices()

        # Это приращения detrended log-prices.
        # Именно этот объект ближе всего к тому, что возвращал старый код.
        self._detrended_log_returns_df = detrended_log_prices_df.diff().dropna()

        return self._detrended_log_returns_df

    def get_prices(self) -> DataFrame:
        # Публичный метод: вернуть цены.
        return self._download_prices().copy()

    def get_log_prices(self) -> DataFrame:
        # Публичный метод: вернуть log-prices.
        return self._build_log_prices().copy()

    def get_log_returns(self) -> DataFrame:
        # Публичный метод: вернуть ОБЫЧНЫЕ log returns.
        # Важно: теперь название метода соответствует математическому смыслу.
        return self._build_log_returns().copy()

    def get_detrended_log_prices(self) -> DataFrame:
        # Публичный метод: вернуть detrended log-prices.
        return self._build_detrended_log_prices().copy()

    def get_detrended_log_returns(self) -> DataFrame:
        # Публичный метод: вернуть detrended log returns.
        return self._build_detrended_log_returns().copy()

    def get_summary(self) -> dict:
        # Краткая сводка по объекту.
        # Полезно для notebook: можно быстро понять, что получилось.
        prices_df: DataFrame = self.get_prices()

        summary: dict = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'input_symbols_count': len(self.symbols),
            'filtered_symbols_count': len(self._get_filtered_symbols()),
            'retained_symbols_count': prices_df.shape[1],
            'filtered_symbols': self._get_filtered_symbols(),
            'retained_symbols': list(prices_df.columns),
            'dropped_symbols_with_missing_history': self._dropped_symbols_with_missing_history,
            'dropped_symbols_with_non_positive_prices': self._dropped_symbols_with_non_positive_prices,
            'prices_shape': prices_df.shape,
        }

        # Если какие-то таблицы уже были построены, добавляем их размеры в summary.
        if self._log_prices_df is not None:
            summary['log_prices_shape'] = self._log_prices_df.shape

        if self._log_returns_df is not None:
            summary['log_returns_shape'] = self._log_returns_df.shape

        if self._detrended_log_prices_df is not None:
            summary['detrended_log_prices_shape'] = self._detrended_log_prices_df.shape

        if self._detrended_log_returns_df is not None:
            summary['detrended_log_returns_shape'] = self._detrended_log_returns_df.shape

        return summary

    def save_all_to_csv(self, output_dir_local: str = '.') -> None:
        # Сохраняем все основные таблицы в CSV.
        # Это удобно для воспроизводимости и для последующего использования в notebook.
        output_dir = Path(output_dir_local)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.get_prices().to_csv(output_dir / 'prices.csv', index=True)
        self.get_log_prices().to_csv(output_dir / 'log_prices.csv', index=True)
        self.get_log_returns().to_csv(output_dir / 'log_returns.csv', index=True)
        self.get_detrended_log_prices().to_csv(output_dir / 'detrended_log_prices.csv', index=True)
        self.get_detrended_log_returns().to_csv(output_dir / 'detrended_log_returns.csv', index=True)


if __name__ == '__main__':
    # Ниже — простой учебный пример использования класса.
    # Он нужен не как финальная исследовательская логика,
    # а как наглядная демонстрация того, как объектом пользоваться.

    # Берём CSV со списком тикеров из текущей папки проекта.
    # csv_path = glob.glob("./nasdaq_screener_*.csv")[-1] # nasdaq_screener_200.csv
    csv_path = glob.glob("./nasdaq_screener_*.csv")[-2] # nasdaq_screener_100.csv


    df_symbols = pd.read_csv(csv_path)
    symbols: list[str] = df_symbols["Symbol"].astype(str).dropna().reset_index(drop=True).tolist()

    log_returns_object: LogReturns = LogReturns(
        symbols_local=symbols,
        start_date_local='2023-01-01',
        end_date_local='2023-10-01',
    )

    # Ниже специально вызываем все основные методы,
    # чтобы наглядно показать полный pipeline объекта.
    prices_df = log_returns_object.get_prices()
    log_prices_df = log_returns_object.get_log_prices()
    log_returns_df = log_returns_object.get_log_returns()
    detrended_log_prices_df = log_returns_object.get_detrended_log_prices()
    detrended_returns_df = log_returns_object.get_detrended_log_returns()

    # При желании можно сохранить весь pipeline целиком.
    log_returns_object.save_all_to_csv('.')

    # Для совместимости со старым сценарием дополнительно сохраним именно detrended returns.
    detrended_returns_df.to_csv('detrended_returns_df_200.csv', index=True)
    print("Detrended returns saved to detrended_returns_df_200.csv")

    # Печатаем summary, чтобы быстро увидеть, что получилось.
    print(log_returns_object.get_summary())
