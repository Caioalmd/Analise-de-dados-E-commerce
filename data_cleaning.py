#Libraries
import os
import sqlite3
import pandas as pd
import numpy as np 
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration and Setup 

# Define the path to the directory containing the Olist dataset
data_path = r'C:\Users\Admin\Desktop\Olist'

# Get a list of all CSV file names in the directory
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]


# Load CSVs into an In-Memory SQLite Database 

# Create a connection to an in-memory SQLite database.
conn = sqlite3.connect(':memory:')
for csv_file in csv_files:
    # Use the file name (without the .csv extension) as the table name
    table_name = os.path.splitext(csv_file)[0]
    file_path = os.path.join(data_path, csv_file)
    
    # Read the CSV file into a temporary DataFrame
    df_temp = pd.read_csv(file_path)
    # Write the DataFrame to a SQL table
    df_temp.to_sql(table_name, conn, index=False, if_exists='replace')
    
    print(f'- Table "{table_name}" created with {df_temp.shape[0]} rows.')

# SQL Practice 
    join_query = '''
SELECT 
    c.*,
    o.*,
    p.*,
    r.*,
    i.*,
    x.*
    
FROM
    olist_customers_dataset c
JOIN 
    olist_orders_dataset o ON c.customer_id = o.customer_id
JOIN 
    olist_order_payments_dataset p ON o.order_id = p.order_id
JOIN
    olist_order_reviews_dataset r ON o.order_id = r.order_id
JOIN
    olist_order_items_dataset i ON o.order_id = i.order_id
JOIN
    olist_products_dataset x ON i.product_id = x.product_id

'''

# Execute the query and load the result directly into a new pandas DataFrame
df = pd.read_sql_query(join_query, conn)


# Close the database connection as it's no longer needed
conn.close()


# Possible columns to be removed by 50%+ null values
null_percentage = df.isna().mean() * 100
null_percentage

# 1. Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

# 2. Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Drop columns that are useless for the analysis
columns_to_drop = ['customer_zip_code_prefix', 'order_approved_at', 'order_delivered_carrier_date',
                  'review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp',
                  'order_item_id', 'review_id', 'shipping_limit_date', 'product_name_lenght',
                  'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm',
                  'product_height_cm', 'product_width_cm', 'order_status', 'seller_zip_code_prefix']  
df = df.drop(columns=columns_to_drop, errors='ignore') 

# 5. Remove columns with only 1 value (no variance)
cols_with_no_variance = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=cols_with_no_variance, errors='ignore')

# 6. Standardize text values
final_cat_cols = df.select_dtypes(include='object').columns
for col in final_cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    
# 7. Date column
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

df.head()

def product_categorization(category):
    """
    Function to map each product category into macro category.
    """
    categories_map = {
        'moveis_escritorio': 'Móveis e Decoração',
        'utilidades_domesticas': 'Casa e Construção',
        'casa_conforto': 'Móveis e Decoração',
        'esporte_lazer': 'Lazer e Entretenimento',
        'informatica_acessorios': 'Eletrodomésticos e Eletrônicos',
        'none': 'Outros',
        'brinquedos': 'Infantil',
        'moveis_decoracao': 'Móveis e Decoração',
        'automotivo': 'Automotivo',
        'climatizacao': 'Eletrodomésticos e Eletrônicos',
        'telefonia': 'Eletrodomésticos e Eletrônicos',
        'beleza_saude': 'Saúde e Beleza',
        'ferramentas_jardim': 'Casa e Construção',
        'pet_shop': 'Pet Shop',
        'cama_mesa_banho': 'Móveis e Decoração',
        'bebes': 'Infantil',
        'relogios_presentes': 'Moda e Acessórios',
        'moveis_cozinha_area_de_servico_jantar_e_jardim': 'Móveis e Decoração',
        'perfumaria': 'Saúde e Beleza',
        'artes': 'Lazer e Entretenimento',
        'papelaria': 'Papelaria e Escritório',
        'fashion_roupa_feminina': 'Moda e Acessórios',
        'consoles_games': 'Eletrodomésticos e Eletrônicos',
        'construcao_ferramentas_iluminacao': 'Casa e Construção',
        'alimentos_bebidas': 'Alimentos e Bebidas',
        'bebidas': 'Alimentos e Bebidas',
        'cool_stuff': 'Outros',
        'fashion_bolsas_e_acessorios': 'Moda e Acessórios',
        'casa_construcao': 'Casa e Construção',
        'malas_acessorios': 'Moda e Acessórios',
        'eletronicos': 'Eletrodomésticos e Eletrônicos',
        'eletrodomesticos_2': 'Eletrodomésticos e Eletrônicos',
        'fashion_roupa_masculina': 'Moda e Acessórios',
        'eletroportateis': 'Eletrodomésticos e Eletrônicos',
        'portateis_casa_forno_e_cafe': 'Eletrodomésticos e Eletrônicos',
        'livros_interesse_geral': 'Lazer e Entretenimento',
        'eletrodomesticos': 'Eletrodomésticos e Eletrônicos',
        'construcao_ferramentas_ferramentas': 'Casa e Construção',
        'sinalizacao_e_seguranca': 'Casa e Construção',
        'instrumentos_musicais': 'Lazer e Entretenimento',
        'construcao_ferramentas_construcao': 'Casa e Construção',
        'musica': 'Lazer e Entretenimento',
        'fashion_calcados': 'Moda e Acessórios',
        'industria_comercio_e_negocios': 'Indústria e Comércio',
        'fashion_underwear_e_moda_praia': 'Moda e Acessórios',
        'dvds_blu_ray': 'Lazer e Entretenimento',
        'construcao_ferramentas_seguranca': 'Casa e Construção',
        'alimentos': 'Alimentos e Bebidas',
        'telefonia_fixa': 'Eletrodomésticos e Eletrônicos',
        'moveis_sala': 'Móveis e Decoração',
        'tablets_impressao_imagem': 'Eletrodomésticos e Eletrônicos',
        'market_place': 'Outros',
        'artigos_de_natal': 'Móveis e Decoração',
        'agro_industria_e_comercio': 'Indústria e Comércio',
        'construcao_ferramentas_jardim': 'Casa e Construção',
        'pcs': 'Eletrodomésticos e Eletrônicos',
        'moveis_quarto': 'Móveis e Decoração',
        'audio': 'Eletrodomésticos e Eletrônicos',
        'livros_importados': 'Lazer e Entretenimento',
        'livros_tecnicos': 'Lazer e Entretenimento',
        'artigos_de_festas': 'Lazer e Entretenimento',
        'portateis_cozinha_e_preparadores_de_alimentos': 'Eletrodomésticos e Eletrônicos',
        'pc_gamer': 'Eletrodomésticos e Eletrônicos',
        'moveis_colchao_e_estofado': 'Móveis e Decoração',
        'la_cuisine': 'Alimentos e Bebidas',
        'flores': 'Móveis e Decoração',
        'fraldas_higiene': 'Infantil',
        'cine_foto': 'Eletrodomésticos e Eletrônicos',
        'cds_dvds_musicais': 'Lazer e Entretenimento',
        'fashion_esporte': 'Moda e Acessórios',
        'casa_conforto_2': 'Móveis e Decoração',
        'artes_e_artesanato': 'Lazer e Entretenimento',
        'fashion_roupa_infanto_juvenil': 'Infantil',
        'seguros_e_servicos': 'Outros'
    }
    return categories_map.get(category, 'Outros') 
df['macro_categoria'] = df['product_category_name'].apply(product_categorization)
df.head()

# Path to the folder where the file will be saved
folder_path = r"C:\Users\Admin\Desktop\Olist"

# File name
file_name = "df_clean.csv"

# Join to create the full path
full_path = os.path.join(folder_path, file_name)

# Save the DataFrame to CSV format
df.to_csv(full_path, index=False, encoding='utf-8-sig')

print(f"✅ Dataset saved successfully at: {full_path}")

