Нужно продолжить развитие signal_backend после этапов 4–5.

Контекст:
- baseline-модели уже реализованы;
- transformer pipeline уже реализован;
- Python-level transformer inference уже есть;
- HTTP API ещё не реализован;
- сейчас нужно сделать:
  1) нормальный inference layer,
  2) FastAPI API поверх него.

Перед началом учти замечания к прошлой работе.

========================
ЧАСТЬ 1. ЗАМЕЧАНИЯ К ПРОШЛОЙ РАБОТЕ
========================

1. Тесты сейчас частично завязаны на Windows-путь `.venv/Scripts/python.exe`.
Это нужно исправить.
Сделай тесты платформонезависимыми:
- использовать `sys.executable`
или другой кроссплатформенный способ запуска subprocess.

2. В CLI-скриптах всё ещё есть ручной `sys.path` injection.
Не нужно ломать существующие команды, но новый код для inference/API не строй вокруг новых path hacks.
Предпочитай чистые импорты из пакета.

3. Текущий inference слой слишком узкий:
- только transformer;
- модель и токенизатор загружаются заново на каждый вызов;
- `batch_predict.py` импортирует приватные функции из `predict.py`.
Это нужно привести в порядок.

4. Не ломай уже работающие этапы:
- inspect_dataset
- make_split
- train_baseline
- train_transformer
- evaluate_model

5. Не делай deployment.
На этом шаге нужен только локальный API-слой приложения.

========================
ЧАСТЬ 2. ЭТАП 6 — INFERENCE LAYER
========================

Цель:
сделать чистый, переиспользуемый inference слой, который:
- умеет работать и с baseline, и с transformer;
- умеет собирать model_input из title/text;
- не загружает модель заново на каждый запрос;
- станет основой для FastAPI.

--------------------------------
2.1. Что нужно добавить в структуру
--------------------------------

Создай/дополни:

src/
  signal_backend/
    inference/
      __init__.py
      artifact_loader.py
      model_input.py
      predictor.py
      predict.py
      batch_predict.py
    serving/
      __init__.py
      schemas.py
      service.py

tests/
  test_inference_predictor.py
  test_model_input.py

Если какие-то текущие файлы нужно не создавать заново, а переработать — перерабатывай.

--------------------------------
2.2. Унифицированный artifact loader
--------------------------------

Нужно сделать единый загрузчик артефактов:
- baseline artifact dir
- transformer artifact dir

Он должен уметь:
- читать `run_summary.json`
- определять `model_type`
- загружать нужные артефакты:
  - для baseline:
    - model.joblib
    - vectorizer.joblib
    - label_mapping.json
  - для transformer:
    - best_model/
    - tokenizer/
    - label_mapping.json
- возвращать нормализованный runtime object

Нужен единый интерфейс вроде:
- load_artifact(artifact_dir: Path) -> LoadedArtifact

Где LoadedArtifact содержит минимум:
- model_type
- label_to_id
- id_to_label
- runtime/model object
- metadata/run_summary

--------------------------------
2.3. Построение model_input
--------------------------------

Сделай явный модуль сборки входа в модель.

Правила:
- если передан `model_input`, использовать его как есть после trim;
- иначе собирать:
  title + "\n\n" + text
- если title пустой и text есть — использовать text;
- если text пустой и title есть — использовать title;
- если оба пустые — ошибка валидации.

Нужна отдельная функция:
- build_model_input(title: str | None, text: str | None, model_input: str | None) -> str

Не дублировать эту логику потом в API-ручках.

--------------------------------
2.4. Predictor abstraction
--------------------------------

Нужно сделать единый predictor layer, который скрывает различия между baseline и transformer.

Нужен общий интерфейс уровня:
- predict_one(...)
- predict_batch(...)

Он должен:
- принимать loaded artifact;
- принимать один или несколько model_input;
- возвращать единый формат результата.

Единый формат результата:
- prediction
- label_id
- scores
- score_type
- label_order
- model_type

Для baseline:
- для logistic regression можно возвращать probabilities;
- для linear SVM — decision scores;
- не притворяться, что decision scores это probabilities.

Для transformer:
- возвращать probabilities из softmax.

--------------------------------
2.5. Кеширование runtime
--------------------------------

Важно:
модель нельзя загружать заново на каждый вызов API.

Нужно сделать service/runtime cache:
- artifact_dir -> loaded artifact
- повторные предсказания используют уже загруженную модель

Требования:
- простая in-memory cache
- без переусложнения
- с понятным кодом

--------------------------------
2.6. Python-level inference API
--------------------------------

Сделай чистый Python API:

- predict_one_from_artifact(
    artifact_dir: Path,
    *,
    title: str | None = None,
    text: str | None = None,
    model_input: str | None = None,
  ) -> dict

- predict_batch_from_artifact(
    artifact_dir: Path,
    items: list[dict],
  ) -> list[dict]

Каждый item в batch может содержать:
- title
- text
- model_input

Ожидаемый выход одного предсказания:
{
  "prediction": "...",
  "label_id": 1,
  "scores": {...} or list[float],
  "score_type": "probabilities" | "decision_scores",
  "label_order": [...],
  "model_type": "tfidf_logreg" | "tfidf_linear_svm" | "transformer_classifier"
}

Предпочтительно отдавать scores как mapping label -> score.
Если это неудобно на внутреннем уровне, хотя бы на внешнем API делай mapping.

--------------------------------
2.7. Обратная совместимость
--------------------------------

Текущие `src/signal_backend/inference/predict.py` и `batch_predict.py` можно переработать, но:
- не оставляй хаотичную смесь старого и нового;
- если меняешь публичные функции — обнови README;
- `scripts/evaluate_model.py` не должен сломаться.

--------------------------------
2.8. Тесты для inference
--------------------------------

Добавь тесты:
- build_model_input работает корректно;
- predictor для baseline возвращает ожидаемую структуру;
- predictor для transformer возвращает ожидаемую структуру;
- batch inference возвращает столько же результатов, сколько входов;
- повторные вызовы используют cache или по крайней мере не ломают runtime.

Критерий готовности этапа 6:
есть чистый переиспользуемый inference layer, который одинаково обслуживает baseline и transformer и может быть вызван как из Python, так и из будущего FastAPI.

========================
ЧАСТЬ 3. ЭТАП 7 — FASTAPI
========================

Цель:
поднять минимальный, чистый HTTP API над inference layer.

--------------------------------
3.1. Что нужно добавить
--------------------------------

Создай/дополни:

apps/
  api/
    main.py

src/
  signal_backend/
    serving/
      __init__.py
      schemas.py
      service.py
      settings.py

configs/
  inference/
    api_config.yaml

tests/
  test_api_smoke.py

Если каких-то каталогов ещё нет — создай.

--------------------------------
3.2. Зависимости
--------------------------------

Добавь в pyproject.toml:
- fastapi
- uvicorn

При необходимости:
- httpx для API-smoke tests
Но только если реально используется.

--------------------------------
3.3. API settings
--------------------------------

Нужен конфиг для API:
- default_artifact_dir
- host
- port
- batch_size default / limits
- maybe max_request_items

Файл:
configs/inference/api_config.yaml

Не захардкоживай artifact dir прямо в коде FastAPI.

--------------------------------
3.4. Pydantic-схемы API
--------------------------------

Нужно сделать запросы/ответы через pydantic модели.

Минимально:

PredictRequest:
- title: optional
- text: optional
- model_input: optional
- artifact_dir: optional

PredictResponse:
- prediction
- label_id
- scores
- score_type
- label_order
- model_type

BatchPredictRequest:
- items: list[PredictRequest-like items]
- artifact_dir: optional

BatchPredictResponse:
- items: list[PredictResponse]

HealthResponse:
- status
- default_artifact_dir

--------------------------------
3.5. Service layer между FastAPI и inference
--------------------------------

FastAPI-ручки не должны напрямую грузить модели и логику.

Сделай service layer:
- resolve artifact dir
- валидировать запрос
- вызывать inference layer
- возвращать сериализуемый ответ

--------------------------------
3.6. Обязательные ручки
--------------------------------

Сделай минимум:

GET /health
POST /predict
POST /batch_predict

GET /health:
- отвечает, что сервис жив;
- показывает configured default artifact dir;
- не должен падать, если модель ещё не загружена.

POST /predict:
- принимает title/text/model_input;
- использует default artifact dir или artifact_dir из запроса;
- возвращает единый prediction response.

POST /batch_predict:
- принимает массив объектов;
- использует ту же схему;
- возвращает массив результатов.

--------------------------------
3.7. Обработка ошибок
--------------------------------

Нужно сделать нормальные HTTP-ошибки:
- 400 для плохого запроса;
- 404 если artifact dir не найден;
- 422 если input невалиден;
- 500 только для реальных внутренних ошибок.

Сообщения должны быть понятными.

--------------------------------
3.8. Пример запуска
--------------------------------

Добавь CLI/README-команду:

python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

Если хочешь, можно добавить:
scripts/run_api.py
Но это не обязательно, если `uvicorn` запуск документирован и работает.

--------------------------------
3.9. Совместимость с фронтендом
--------------------------------

API должен быть готов для signal_front.

Поэтому внешний формат ответа должен быть стабильным и понятным:
- prediction
- scores
- label_order
- model_type

Не отдавай внутренние объекты sklearn/torch наружу.

--------------------------------
3.10. API smoke tests
--------------------------------

Добавь минимальные тесты:
- /health отвечает 200
- /predict отвечает 200 на валидный запрос
- /batch_predict отвечает 200 на валидный пакет
- невалидный пустой запрос даёт корректную ошибку

Тесты должны быть по возможности лёгкими.
Не тянуть большое обучение внутри API тестов.

--------------------------------
3.11. README
--------------------------------

Обнови README:
- теперь этап inference layer реализован;
- теперь есть FastAPI;
- покажи примеры запросов:
  - /predict
  - /batch_predict
- покажи как запускать API.

--------------------------------
3.12. Критерий готовности этапа 7
--------------------------------

После этапа 7 должно быть возможно:
1. загрузить готовый artifact;
2. локально запускать FastAPI;
3. получить предсказание по одному тексту;
4. получить предсказания пачкой;
5. использовать это как основу для signal_front.

========================
ЧАСТЬ 4. ОБЩИЕ ТРЕБОВАНИЯ
========================

1. Не ломать baseline и transformer training.
2. Не ломать scripts/evaluate_model.py.
3. Не дублировать логику сборки model_input в нескольких местах.
4. Не грузить модель заново на каждый запрос.
5. Не смешивать HTTP-схемы с низкоуровневым inference кодом.
6. Не делать deployment / docker / reverse proxy на этом этапе.
7. Не оставлять старые legacy-функции, если они уже полностью заменены новой архитектурой.

========================
ЧТО НУЖНО ПОКАЗАТЬ В ФИНАЛЬНОМ ОТВЕТЕ
========================

После выполнения задачи покажи:
1. какие замечания к прошлой работе были исправлены;
2. какие файлы созданы/изменены;
3. как теперь устроен inference layer;
4. как запускается FastAPI;
5. какие ручки доступны;
6. пример запроса и ответа для /predict;
7. какие шаги останутся следующими после этого.

Критерий успеха:
signal_backend получает чистый inference layer и минимальный рабочий FastAPI-сервис поверх уже реализованных baseline и transformer pipelines.