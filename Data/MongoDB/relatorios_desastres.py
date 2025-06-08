import csv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

def export_collection_to_csv(collection, filename):
    """
    Exporta uma coleção do MongoDB para um arquivo CSV, excluindo o campo _id
    
    Args:
        collection: Coleção do MongoDB
        filename: Nome do arquivo CSV de saída
    """

    documents = list(collection.find()) 
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        if len(documents) > 0:

            first_doc = documents[0]
            fieldnames = [key for key in first_doc.keys() if key != '_id']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            
            for doc in documents:
                row = {k: v for k, v in doc.items() if k != '_id'}
                writer.writerow(row)
            
            print(f"Exportados {len(documents)} documentos para {filename}")
        else:
            print(f"A coleção está vazia, nenhum dado exportado para {filename}")

def main():

    mongo_uri = "mongodb://localhost:27017/"  
    db_name = "Global_Solution"              
    collection1_name = "relatorios_desastres"  
    
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        print(f"Conectado ao MongoDB: {mongo_uri}")
        print(f"Banco de dados: {db_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file1 = f"{collection1_name}_{timestamp}.csv"
        
        print(f"\nExportando coleção: {collection1_name}")
        export_collection_to_csv(db[collection1_name], csv_file1)
        
    except Exception as e:
        print(f"Erro ao conectar ou exportar dados: {e}")
    finally:
        client.close()
        print("\nConexão com o MongoDB fechada.")

if __name__ == "__main__":
    main()