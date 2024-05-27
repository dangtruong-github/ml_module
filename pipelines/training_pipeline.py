from steps.Service import Service 
from steps.data_retrieving import retrieve_data

def pipeline(): 
    df_movie, df_user, df_rating = retrieve_data()
    service = Service(df_rating)
    service.train()
    service.test()
    return service.predict()

if "__name__" == "__main__":
    pipeline(df)