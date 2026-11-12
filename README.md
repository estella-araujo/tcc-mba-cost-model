# Análise de custos de serviços em nuvem

Repositório de código e dados reproduzíveis do projeto de comparação de custos entre AWS, Microsoft Azure e Google Cloud entregue como Trabalho de Conclusão de Curso para o MBA em Data Science.

## Conteúdo

- Scripts Python para coleta, geração de workloads, preparação de dados e análises.
- Configurações de consulta sem credenciais. A chave do Infracost deve ser fornecida pela variável de ambiente `INFRACOST_API_KEY`.
- Dados de preços e workloads em Parquet, dados processados e resultados tabulares/gráficos.

## Dependências

```powershell
python -m pip install -r requirements.txt
```

## Atenção

Antes de executar a coleta, defina `INFRACOST_API_KEY` no ambiente. O `.gitignore` exclui arquivos `.env` e configurações locais.

## Execução

Os scripts aceitam caminhos de entrada e saída por argumentos de linha de comando. Consulte `--help` para as opções de cada etapa. As saídas de coleta e processamento são gravadas localmente e não substituem os arquivos da pasta original.
