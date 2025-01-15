import polars as pl
from tqdm import tqdm

total_linhas = 1_000_000_000  # Total de linhas conhecido
chunksize = 100_000_000  # Define o tamanho do chunk
filename = "data/medicoes_1000000000.txt"  # Certifique-se de que este é o caminho correto para o arquivo

def process_chunk(chunk):
    # Agrega os dados dentro do chunk usando Polars
    aggregated = chunk.group_by('station').agg([
        pl.col('measure').min().alias('min'),
        pl.col('measure').max().alias('max'),
        pl.col('measure').mean().alias('mean'),
    ])
    return aggregated

def create_df_with_polars(filename, total_linhas, chunksize=chunksize):
    total_chunks = total_linhas // chunksize + (1 if total_linhas % chunksize else 0)
    results = []

    # Usando scan_csv para carregar em streaming
    df = pl.scan_csv(
        filename,
        separator=';',  # Substituto para "delimiter"
        has_header=False,
        schema_overrides={'station': pl.Utf8, 'measure': pl.Float64}  # Atualizado para schema_overrides
    )

    # Processa os dados em chunks
    for i in tqdm(range(total_chunks), desc="Processando"):
        start = i * chunksize
        end = start + chunksize
        chunk = df.slice(start, chunksize).collect()

        if chunk.height == 0:  # Final da leitura
            break

        results.append(process_chunk(chunk))

    final_df = pl.concat(results, how="vertical")

    final_aggregated_df = final_df.group_by('station').agg([
        pl.col('min').min(),
        pl.col('mean').mean(),
        pl.col('max').max(),
    ])

    return final_aggregated_df

if __name__ == "__main__":
    import time

    print("Iniciando o processamento do arquivo.")
    start_time = time.time()
    df = create_df_with_polars(filename, total_linhas, chunksize)
    took = time.time() - start_time

    print(df.head())
    print(f"Processing took: {took:.2f} sec")
