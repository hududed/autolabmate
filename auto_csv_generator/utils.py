from decimal import Decimal, ROUND_HALF_UP
from typing import List, Union


def round_to_nearest(value: Union[int, float], interval: float) -> Union[int, float]:
    interval_decimal = Decimal(str(interval))
    if isinstance(value, int):
        rounded_value = (Decimal(value) / interval_decimal).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ) * interval_decimal
        return int(rounded_value)
    value_decimal = Decimal(str(value))
    rounded_value = (value_decimal / interval_decimal).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * interval_decimal
    return float(rounded_value)


def format_values(
    values: List[List[Union[int, float, str]]], decimal_places: int
) -> List[List[Union[int, float, str]]]:
    format_str = f"{{:.{decimal_places}f}}"
    formatted_values = []
    for row in values:
        formatted_row = [
            format_str.format(value) if isinstance(value, float) else value
            for value in row
        ]
        formatted_values.append(formatted_row)
    return formatted_values
