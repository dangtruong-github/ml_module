from steps.Service import Service 
from steps.data_retrieving import retrieve_data

def pipeline(): 
    df_movie, df_user, df_rating = retrieve_data()
    service = Service(data=df_rating, df_user=df_user, df_movie=df_movie)
    service.train()
    service.test()
    return service.predict()

if "__name__" == "__main__":
    pipeline()