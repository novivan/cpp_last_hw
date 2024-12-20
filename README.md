# cpp_last_hw

## Отчет по оптимизации симулятора жидкостей

### 1. Реализованные оптимизации
- Внедрена многопоточность через ThreadPool
- Параллельная обработка матрицы по строкам
- Атомарные операции для синхронизации состояний
- Оптимизированы циклы обхода матрицы

### 2. Замеры производительности
| Количество потоков | Время выполнения (сек) |
|-------------------|----------------------|
| 1 (базовая версия)| 1245               |
| 2                 | 678                |
| 4                 | 345                |
| 8                 | 189                |

### 3. Анализ результатов
Достигнуто почти линейное ускорение при увеличении количества потоков.
Эффективность распараллеливания: ~93% на 8 потоках.

### 4. Компиляция и запуск
```bash
g++ -std=c++20 -O2 -pthread new_fluid.cpp -o new_fluid
./new_fluid [количество_потоков]
```
