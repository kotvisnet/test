import pandas as pd
import numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

places = [
    ("Москва", "Тверской район", 55.76, 37.61),
    ("Москва", "Раменки", 55.70, 37.51),
    ("Москва", "Митино", 55.85, 37.36),
    ("Московская область", "Балашиха", 55.80, 37.94),
    ("Московская область", "Химки", 55.89, 37.43),
    ("Московская область", "Мытищи", 55.91, 37.73),
    ("Санкт-Петербург", "Адмиралтейский район", 59.93, 30.31),
    ("Санкт-Петербург", "Выборгский район", 60.02, 30.31),
    ("Краснодарский край", "Краснодар", 45.04, 38.97),
    ("Краснодарский край", "Сочи", 43.60, 39.73),
    ("Республика Татарстан", "Казань", 55.79, 49.12),
    ("Республика Татарстан", "Набережные Челны", 55.74, 52.40),
    ("Новосибирская область", "Новосибирск", 55.03, 82.92),
    ("Свердловская область", "Екатеринбург", 56.84, 60.61),
    ("Приморский край", "Владивосток", 43.12, 131.88),
    ("Хабаровский край", "Хабаровск", 48.48, 135.07),
]

rows = []
years = range(2014, 2025)

np.random.seed(42)

for region, municipality, lat, lon in places:
    population = np.random.randint(80_000, 1_700_000)
    trend = np.random.uniform(-0.018, 0.026)

    for year in years:
        population = int(population * (1 + trend + np.random.normal(0, 0.006)))
        birth_rate = round(np.random.uniform(7, 14), 2)
        death_rate = round(np.random.uniform(8, 16), 2)
        natural_growth = round(birth_rate - death_rate, 2)
        migration = int(np.random.normal(1200, 5000))
        density = round(population / np.random.uniform(50, 900), 2)

        rows.append({
            "year": year,
            "region": region,
            "municipality": municipality,
            "population": population,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "natural_growth": natural_growth,
            "migration": migration,
            "density": density,
            "lat": lat,
            "lon": lon
        })

df = pd.DataFrame(rows)
df.to_csv("data/population.csv", index=False, encoding="utf-8-sig")

print("Готово: data/population.csv")