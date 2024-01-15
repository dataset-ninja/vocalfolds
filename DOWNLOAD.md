Dataset **Laryngeal Endoscopic** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/e/2/BQ/ZhE4To5xYRVA2nOblmz0YIlsuV8mj0yGD0lixT0OrAAj3nR29VpFleD1jalPCSmH5Pfu7IUGNLw4eSjPttnxWLZJtfp8qoeJC2OJcQbQRuWAfzLJi9dOZQgLJ1Mp.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Laryngeal Endoscopic', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://github.com/imesluh/vocalfolds).