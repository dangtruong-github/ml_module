from steps.Service import Service 

def pipeline(df): 
    service = Service(df)
    service.train()
    service.test()
    return service.predict()

if "__name__" == "__main__":
    pipeline(df)