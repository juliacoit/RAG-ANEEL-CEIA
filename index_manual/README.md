# Index Manual

Esta pasta foi criada para armazenar arquivos de indexação gerados manualmente ou em ambientes externos (como Google Colab com GPU).

### Requisito: `embeddings_bge_m3.npy`

Para que a indexação manual funcione, esta pasta **precisa** conter o arquivo `embeddings_bge_m3.npy`. 
Este arquivo contém os vetores (embeddings) pré-calculados para os chunks do projeto, utilizando o modelo `BAAI/bge-m3`.

**Nota:** Devido ao tamanho do arquivo `.npy`, ele não é versionado no Git por padrão. Certifique-se de baixar ou gerar este arquivo e colocá-lo aqui antes de rodar scripts que dependam de `index_manual`.
