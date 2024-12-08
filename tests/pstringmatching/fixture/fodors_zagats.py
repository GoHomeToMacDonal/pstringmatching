import pytest
import csv


@pytest.fixture
def fodors():
    with open("tests/fodors.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, quotechar="'")
        data = list(reader)
        columns = data[0]
        data = data[1:]
        items = []
        for row in data:
            if len(row) == len(columns):
                items.append({columns[i]: row[i] for i in range(len(columns))})
        return items


@pytest.fixture
def zagats():
    with open("tests/zagats.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, quotechar="'")
        data = list(reader)
        columns = data[0]
        data = data[1:]
        items = []
        for row in data:
            if len(row) == len(columns):
                items.append({columns[i]: row[i] for i in range(len(columns))})
        return items
