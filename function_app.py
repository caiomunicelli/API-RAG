import azure.functions as func
from RAG import main

app = func.FunctionApp()

@app.function_name(name="RAG")
@app.route(route="rag", auth_level=func.AuthLevel.ANONYMOUS)
def rag_function(req: func.HttpRequest) -> func.HttpResponse:
    return main(req)
