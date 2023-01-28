import mlflow

def calculate_nth_power(x,y):
    return x**y


if __name__== '__main__':
    with mlflow.start_run():
        x, n = 3,5
        y = calculate_nth_power(x,n)
        mlflow.log_param("x",x)
        mlflow.log_param("n",n)
        mlflow.log_param("y",y)



