from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.utils import (
    load_dataset, build_etl_pipeline,
    evaluate_regression, evaluate_classification,
    plot_regression_all_metrics, plot_regression_winner_scatter, plot_regression_residuals,
    plot_classification_metrics, plot_confusion_matrix, plot_roc_curve_binary
)
from app.models import REGRESSION_MODELS, CLASSIFICATION_MODELS

app = FastAPI(title="Minería de Datos - Evaluación de Modelos")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    file: UploadFile | None = None,
    url: str | None = Form(default=None),
    target: str = Form(...),
    problem_type: str = Form(...),
):
    try:
        file_bytes = await file.read() if file is not None else None
        df = load_dataset(file_bytes, url)
        preprocessor, X, y = build_etl_pipeline(df, target)

        if problem_type == "regression":
            results, winner, store = evaluate_regression(REGRESSION_MODELS, preprocessor, X, y)

            metrics_img = plot_regression_all_metrics(results)
            scatter_img = plot_regression_winner_scatter(store["y_test"], store["preds"], title=f"{winner}: y_real vs y_pred")
            resid_img = plot_regression_residuals(store["y_test"], store["preds"])

            context = {
                "request": request,
                "problem_type": "Regresión",
                "target": target,
                "results": results,
                "winner": winner,
                "images": {
                    "metrics": metrics_img,
                    "scatter": scatter_img,
                    "residuals": resid_img
                }
            }
            return templates.TemplateResponse("results.html", context)

        elif problem_type == "classification":
            results, winner, store = evaluate_classification(CLASSIFICATION_MODELS, preprocessor, X, y)
            metrics_img = plot_classification_metrics(results)
            cm_img = plot_confusion_matrix(store["y_test"], store["preds"])
            roc_img = None
            if store["is_binary"]:

                # Aquí simplificamos repitiendo split, en un proyecto grande guardarlo antes.
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                roc_img = plot_roc_curve_binary(store["pipe"], X_test, y_test)

            context = {
                "request": request,
                "problem_type": "Clasificación",
                "target": target,
                "results": results,
                "winner": winner,
                "images": {
                    "metrics": metrics_img,
                    "confusion": cm_img,
                    "roc": roc_img
                }
            }
            return templates.TemplateResponse("results.html", context)
        else:
            return templates.TemplateResponse("upload.html", {"request": request, "error": "Tipo de problema inválido"})

    except Exception as e:
        return templates.TemplateResponse("upload.html", {"request": request, "error": str(e)})
