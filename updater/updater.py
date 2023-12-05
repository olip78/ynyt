import os
import pandas as pd

import uvicorn
from fastapi import FastAPI, Request, HTTPException


#os.environ['PATH_DATA'] = '../data/'
PATH_DATA = os.environ['PATH_DATA']

app = FastAPI()

# endpoint to receive predictions
@app.post("/update/{period}")
async def update_data(period: str, request: Request) -> dict:
    """Gets and save prediction results
    Arguments:
      period : str
        either "day" or "month"
      request : Request
        request.json is jsonized pd.DataFrame
    Returtns:
      status response : dict
    """
    try:
        data = await request.json()
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(PATH_DATA, 'results', f'results_{period}.csv'), index=False)
        return {"status": f'{period} predictions is updated'}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


# endpoint for first update
@app.get("/month/")
async def first_update(request: Request) -> dict:
    """returns 0 if results_month.csv is there, 1 otherwise
    """
    file_path = os.path.join(PATH_DATA, 'results', 'results_month.csv')

    if os.path.exists(file_path):
        result = 0
    else:
        result = 1

    return {"result": result}

# healthcheck endpoint
@app.get('/healthcheck')
async def check():
    return {'status': 'OK'}

# start the FastAPI server‚
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8081)
