# Identificação de Cena Acústica utilizando Classificação Hierárquica

Sistema de identificação de cena acústico desenvolvido para um projeto de iniciação científica, sob a orientação do prof. Yandre Costa, na Universidade Estadual de Maringá.

## Execução

1. Baixe o dataset [TAU Urban Acoustic Scenes 2019, Development dataset](https://zenodo.org/records/2589280)
   e o salve na pasta raiz do projeto.

2. Certifique-se que possui Python 3.11 instalado.

3. Crie um venv:
   ```sh
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   ```

3. Numa linha de comando, instale os pacotes dependentes:

   ```sh
   $ pip install -r requirements.txt
   ```

4. Altere a varíavel `DEV_DS` no arquivo `codigo/main.py` para apontar para o 
   dataset baixado.

5. Defina o vetor de características a ser utilizado na função `extract_features`
   do arquivo `codigo/main.py`. Os vetores disponíveis podem ser vistos no arquivo
   `codigo/feature_vector.py`.

6. Execute o classificador.

   ```sh
   $ python3 codigo/main.py
   ```
## Licença

A porção do código escrita pelo autor está licenciada sob a GPLv3. Mais detalhes no arquivo `LICENSE`.