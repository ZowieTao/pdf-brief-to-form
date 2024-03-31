## Config 

config your env：copy the `.env.template` to `.env`

```shell
cp -r ./.env.template ./.env
```

## Start

```shell
# create python env
conda create -n pdf-brief-to-form python=3.10
conda activate pdf-brief-to-form

# install 
pip install black mypy pylint
pip install -r requirements.txt
pip install -e .

# run
python -m pdf_brief_to_form
```

