# Map Image Check

Поиск изображений топографических и terrain-карт среди локальных, сетевых и удалённых файлов.

**Версия:** 1.1.0  
**Автор:** av_nefedov

## Возможности

- **Сканирование** локальных дисков, выбранных папок и удалённых компьютеров через AD (UNC-шары `C$`, `D$` и т.д.)
- **Гибридная классификация:** эвристика OpenCV → ML-модель (logistic regression) → опционально LLM (Ollama) для «серой зоны»
- **База данных SQLite** — сохранение найденных карт, предпросмотр, метки «Это карта» / «Не карта» для дообучения модели
- **Просмотрщик базы** без повторного сканирования; при скане уже известные файлы пропускаются
- **Фильтры в базе:** по имени компьютера и по ML-оценке (`> 90%`, `80–90%`)
- **LLM-анализ** выбранной карты через Ollama (`llama3.2-vision` и др.)

## Требования

- Python 3.10+
- Windows (GUI, AD, UNC-пути)
- Для LLM: [Ollama](https://ollama.com/) с vision-моделью

## Установка

```bash
git clone https://github.com/avnefedov1-hub/map-image-check.git
cd map-image-check
pip install -r map_image_check/requirements.txt
```

## Запуск

Из корня репозитория:

```bash
python -m map_image_check.gui_app
```

или:

```bash
python map_image_check_app.py
```

## Тесты

```bash
python -m unittest map_image_check.test_hybrid_pipeline map_image_check.test_ad_remote map_image_check.test_llm_analysis map_image_check.test_image_store_viewer -v
```

## Горячие клавиши

| Клавиши | Действие |
|---------|----------|
| `Ctrl+B` | Открыть базу данных |
| `Ctrl+,` | Настройки |
| `Delete` | Удалить выбранную запись из БД |

## Данные на диске

| Файл | Назначение |
|------|------------|
| `map_image_check.sqlite3` | База найденных карт (создаётся автоматически) |
| `map_classifier.joblib` | Обученная ML-модель |
| `map_scan_results.csv` | Отчёт последнего сканирования |

Эти файлы не попадают в git (см. `.gitignore`).

## Сборка

В репозитории есть скрипты PyInstaller (`MapImageCheck.spec`) и Inno Setup (`build_assets/MapImageCheckInstaller.iss`).

## Лицензия

Проект распространяется as-is для внутреннего использования. Уточните лицензию у автора при публикации.
