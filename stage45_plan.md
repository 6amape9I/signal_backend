Нужно продолжить развитие репозитория signal_backend.

Контекст:
- первые 3 этапа уже реализованы:
  1) каркас проекта,
  2) импорт и валидация JSONL-датасета,
  3) воспроизводимый stratified split.
- текущая база уже рабочая:
  - есть README,
  - pyproject.toml,
  - paths.py,
  - схемы,
  - загрузка JSONL,
  - валидация,
  - label mapping,
  - split,
  - CLI-скрипты,
  - тесты.
- локальный датасет по-прежнему должен лежать в:
  data/input/dataset_clean.jsonl
- на этом следующем шаге нужно реализовать:
  4) baseline-модели
  5) transformer-модель

Сначала учти замечания к прошлой работе.

========================
ЧАСТЬ 1. ЗАМЕЧАНИЯ К ПРОШЛОЙ РАБОТЕ
========================

Перед началом новых этапов нужно аккуратно улучшить текущую базу:

1. Начни использовать configs/data/dataset_config.yaml как реальный источник дефолтных параметров.
Сейчас конфиг существует, но CLI и split-логика живут в основном на хардкодах.
Нужно:
- читать dataset path / split defaults из dataset_config.yaml;
- оставлять CLI-аргументы как override поверх конфига;
- не ломать текущий UX.

2. Не ломай совместимость текущих команд:
- python scripts/inspect_dataset.py
- python scripts/make_split.py
Они должны продолжать работать.

3. Не переписывай ядро этапов 1–3 без необходимости.
Нужно строить следующее поверх уже работающей структуры.

4. По возможности не держи pytest как runtime-обязательную зависимость.
Если удобно и безопасно — переведи pytest в optional/dev dependency.
Но это вторично. Если будет мешать основным этапам, не трать на это время.

5. Не меняй расположение локального датасета:
- data/input/dataset_clean.jsonl

========================
ЧАСТЬ 2. ЭТАП 4 — BASELINE-МОДЕЛИ
========================

Нужно добавить в проект baseline-уровень для честного сравнения с transformer-моделью.

Цель:
получить быстрые и воспроизводимые базовые модели для классификации по полю `model_input` с целевой меткой `category_teacher_final`.

--------------------------------
2.1. Что нужно добавить в структуру
--------------------------------

Создай/дополни структуру:

configs/
  train/
    baseline_tfidf_logreg.yaml
    baseline_tfidf_linear_svm.yaml

src/
  signal_backend/
    baselines/
      __init__.py
      tfidf_features.py
      train_logreg.py
      train_linear_svm.py
    training/
      __init__.py
      metrics.py
      evaluate.py
      save_artifacts.py

scripts/
  train_baseline.py
  evaluate_model.py

tests/
  test_baseline_training.py
  test_metrics.py

--------------------------------
2.2. Общие правила для baseline-этапа
--------------------------------

Использовать:
- входной текст: model_input
- target: category_teacher_final

Не использовать как признаки:
- project
- project_nick
- type
- publish_date
- fronturl
- picture
- badge

Сначала обучаемся на split-файлах:
- data/processed/train.jsonl
- data/processed/val.jsonl
- data/processed/test.jsonl

Если split-файлы отсутствуют:
- можно либо падать с понятной ошибкой,
- либо предложить сначала запустить scripts/make_split.py
Но не делать скрытую магию без явного сообщения.

--------------------------------
2.3. Baseline #1 — TF-IDF + Logistic Regression
--------------------------------

Нужно реализовать baseline-модель:
- TF-IDF vectorizer
- Logistic Regression classifier

Требования:
- конфиг через YAML
- возможность задавать:
  - max_features
  - ngram_range
  - min_df
  - max_df
  - lowercase
  - class_weight
  - C
  - max_iter
  - random_state
- class_weight должен поддерживать:
  - null / none
  - "balanced"

--------------------------------
2.4. Baseline #2 — TF-IDF + Linear SVM
--------------------------------

Нужно реализовать вторую baseline-модель:
- TF-IDF vectorizer
- Linear SVM classifier

Требования:
- отдельный YAML-конфиг
- параметры vectorizer аналогично baseline #1
- параметры модели:
  - C
  - class_weight
  - random_state

Важно:
если используешь модель без нормальных вероятностей, API baseline-предсказаний всё равно должен работать предсказуемо:
- либо возвращать decision scores,
- либо явно помечать, что это scores, а не calibrated probabilities.

Не делай фальшивые вероятности.

--------------------------------
2.5. Метрики и оценка
--------------------------------

Нужно реализовать единый слой метрик.

Считать минимум:
- accuracy
- macro_f1
- weighted_f1
- per-class precision
- per-class recall
- per-class f1
- confusion matrix

Нужно отдельно оценивать:
- val
- test

Сохранять в артефакты:
- metrics.json
- classification_report.json
- confusion_matrix.csv

--------------------------------
2.6. Формат артефактов baseline
--------------------------------

Сделай единый формат сохранения запусков.

Для каждого baseline-запуска сохраняй в:
data/artifacts/<run_name>/

Минимально:
- metrics.json
- classification_report.json
- confusion_matrix.csv
- label_mapping.json
- config_snapshot.yaml
- model.joblib
- vectorizer.joblib
- run_summary.json

run_summary.json должен содержать:
- model_type
- dataset_path
- train_path
- val_path
- test_path
- row_counts
- classes
- main metrics

--------------------------------
2.7. CLI для baseline
--------------------------------

Нужно сделать scripts/train_baseline.py

CLI должен:
- принимать путь к конфигу
- принимать model type:
  - tfidf_logreg
  - tfidf_linear_svm
- загружать split-файлы
- обучать модель
- оценивать на val и test
- сохранять артефакты
- печатать summary в консоль

Примеры ожидаемого запуска:
- python scripts/train_baseline.py --config configs/train/baseline_tfidf_logreg.yaml
- python scripts/train_baseline.py --config configs/train/baseline_tfidf_linear_svm.yaml

Нужно также сделать scripts/evaluate_model.py для повторной оценки уже сохранённой модели на test-части.

--------------------------------
2.8. Тесты для baseline
--------------------------------

Добавь минимальные тесты:
- baseline можно обучить на маленьком toy dataset
- prediction возвращает нужное число результатов
- metrics считаются без ошибок
- label mapping согласован
- train/val/test pipeline не ломается

--------------------------------
2.9. Критерий готовности этапа 4
--------------------------------

После этапа 4 должно быть возможно:
1. сделать split,
2. обучить две baseline-модели,
3. получить воспроизводимые метрики,
4. сохранить артефакты,
5. сравнить результаты baseline между собой.

========================
ЧАСТЬ 3. ЭТАП 5 — TRANSFORMER-МОДЕЛЬ
========================

Нужно добавить первый рабочий transformer pipeline.

Цель:
обучить text classifier по `model_input` -> `category_teacher_final` и сравнить его с baseline.

--------------------------------
3.1. Что нужно добавить в структуру
--------------------------------

Создай/дополни:

configs/
  train/
    transformer_classifier.yaml

src/
  signal_backend/
    models/
      __init__.py
      transformer_classifier.py
    training/
      train_transformer.py
      dataset_adapter.py
      early_stopping.py
      checkpointing.py
    inference/
      __init__.py
      predict.py
      batch_predict.py

scripts/
  train_transformer.py

tests/
  test_transformer_smoke.py

--------------------------------
3.2. Общие правила для transformer-этапа
--------------------------------

Использовать:
- вход: model_input
- target: category_teacher_final

Не использовать metadata как признаки.

Нужна конфигурация через YAML.

Минимально конфиг должен поддерживать:
- model_name_or_path
- max_length
- batch_size
- learning_rate
- weight_decay
- num_epochs
- warmup_ratio или warmup_steps
- random_seed
- gradient_accumulation_steps
- class_weight mode (optional)
- early_stopping patience
- output_dir / run_name

--------------------------------
3.3. Dataset adapter
--------------------------------

Нужно сделать адаптер датасета для transformer training:
- берёт split DataFrame / записи
- токенизирует поле model_input
- возвращает input_ids, attention_mask, label_id
- использует label mapping из train split

Важное требование:
- label mapping должен строиться на train split и быть тем же самым для val/test/inference.

--------------------------------
3.4. Модель
--------------------------------

Нужна одна простая и понятная реализация:
- Hugging Face AutoTokenizer
- Hugging Face AutoModel / AutoModelForSequenceClassification
или свой lightweight wrapper, если это аккуратно

Не нужно делать зоопарк моделей.
Нужен один чистый training path.

--------------------------------
3.5. Обучение
--------------------------------

Нужно реализовать:
- train loop
- validation after each epoch
- сохранение best checkpoint по val macro_f1
- логирование train/val metrics
- early stopping
- reproducibility через seed

Поддержать:
- CPU
- CUDA, если доступна

Если CUDA недоступна — не падать, а обучаться на CPU.

--------------------------------
3.6. Метрики transformer
--------------------------------

Считать те же ключевые метрики, что и у baseline:
- accuracy
- macro_f1
- weighted_f1
- classification report
- confusion matrix

Оценивать на:
- val
- test

--------------------------------
3.7. Формат артефактов transformer
--------------------------------

Сохранять в:
data/artifacts/<run_name>/

Минимально:
- metrics.json
- classification_report.json
- confusion_matrix.csv
- label_mapping.json
- config_snapshot.yaml
- train_log.jsonl
- tokenizer/
- best_model/
- run_summary.json

run_summary.json должен включать:
- model_type = transformer_classifier
- base_model_name
- dataset paths
- split sizes
- classes
- best validation metric
- final test metrics

--------------------------------
3.8. CLI для transformer
--------------------------------

Нужно сделать scripts/train_transformer.py

CLI должен:
- принимать путь к YAML-конфигу
- загружать split-файлы
- строить label mapping
- обучать transformer
- сохранять best checkpoint
- считать тестовые метрики
- печатать summary в консоль

Ожидаемый запуск:
- python scripts/train_transformer.py --config configs/train/transformer_classifier.yaml

--------------------------------
3.9. Inference-ready слой
--------------------------------

Даже если полноценный API ещё не делаем, подготовь инференс-слой:
- функция predict_one(title: str, text: str) или predict_one(model_input: str)
- функция predict_batch(...)
- загрузка best_model + tokenizer + label mapping

Важно:
это пока не HTTP API.
Это только Python-level inference utilities, чтобы потом на них оперся FastAPI.

--------------------------------
3.10. Smoke test для transformer
--------------------------------

Добавь хотя бы один очень лёгкий smoke test:
- tiny synthetic dataset
или
- mock/small-model path, если это безопаснее

Тест не должен тянуть тяжёлое обучение на долгие минуты.
Нужно просто проверить, что pipeline поднимается и не ломается архитектурно.

--------------------------------
3.11. Критерий готовности этапа 5
--------------------------------

После этапа 5 должно быть возможно:
1. обучить baseline,
2. обучить transformer,
3. сравнить их по одинаковым метрикам,
4. получить артефакты,
5. иметь Python inference utilities для следующего этапа API.

========================
ЧАСТЬ 4. ОБЩИЕ ТРЕБОВАНИЯ К РЕАЛИЗАЦИИ
========================

1. Не ломать этапы 1–3.
2. Не использовать dataset_clean.jsonl из корня проекта.
3. Работать через data/input и data/processed.
4. Все пути через pathlib.
5. Все основные параметры через YAML-конфиги.
6. Не делать магию без логов.
7. Не плодить несколько разных форматов артефактов.
8. Не делать одновременно API и training — сейчас только baseline и transformer.
9. README нужно обновить так, чтобы:
   - были описаны новые команды обучения;
   - было понятно, где baseline, где transformer;
   - было ясно, какие этапы уже реализованы.

========================
ЧТО НУЖНО ПОКАЗАТЬ В ФИНАЛЬНОМ ОТВЕТЕ
========================

После выполнения задачи покажи:
1. какие замечания из прошлой работы были исправлены;
2. какие новые файлы созданы;
3. какие baseline-модели добавлены;
4. как запускать baseline;
5. как запускать transformer;
6. какой единый формат артефактов теперь используется;
7. какие шаги останутся следующими после этого (например, inference API / frontend integration).

Критерий успеха:
signal_backend получает полноценный training layer:
- с двумя baseline-моделями,
- с одной transformer-моделью,
- с единым набором метрик,
- с воспроизводимыми артефактами,
- без слома ранее реализованных этапов импорта, валидации и split.