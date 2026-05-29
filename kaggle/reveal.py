# import torch
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim

# class ClassificationNN(nn.Module):
#     def __init__(self, inputSize):
#         super(ClassificationNN, self).__init__()
#         self.fc1 = nn.Linear(inputSize, 34)
#         self.fc2 = nn.Linear(34, 68)
#         self.fc3 = nn.Linear(68, 136)
#         self.dropout = nn.Dropout(0.25)
#         self.fc4 = nn.Linear(136, 68)
#         self.fc5 = nn.Linear(68, 2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc4(x))
#         x = self.relu(self.fc5(x))

#         return x

# inputSize = 12 # will change it later

# model = ClassificationNN(inputSize)

# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
# criterion = nn.SmoothL1Loss()

# num_epochs = 200
# dummy_train_tensor = torch.randn(10, inputSize) # Example dummy input
# dummy_target_tensor = torch.randint(0, 2, (10,)).float().unsqueeze(1) # Example dummy target for binary classification

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model.forward(dummy_train_tensor) # Used dummy data
#     loss = criterion(output[:, 0], dummy_target_tensor.squeeze())
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch + 1} / {num_epochs}: loss {loss.item():.4f}")

# # Replaced 30 with range(30) and dummy data for now
# dummy_test_tensor = torch.randn(5, inputSize) 
# dummy_test_target_tensor = torch.randint(0, 2, (5,)).float().unsqueeze(1)

# for epoch in range(30):
#     model.eval() 
#     with torch.no_grad(): 
#         output = model.forward(dummy_test_tensor)
#         loss = criterion(output[:, 0], dummy_test_target_tensor.squeeze())

#     print(f"Test Epoch {epoch + 1}: loss {loss.item():.4f}")


TRAIN = '../data/real-vs-fake-faces-stylegan3'

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import arrow
import base64
import pandas as pd
import tqdm
import os

from glob import iglob
from io import BytesIO
from os.path import basename
from os.path import isdir
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool

from bokeh.plotting import figure
from bokeh.plotting import output_notebook
from bokeh.plotting import show
from bokeh.palettes import Turbo256
from bokeh.transform import factor_cmap

DEVICE = torch.device('cpu')
OUTPUT_SIZE = 2048

model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

extraction_layer = model._modules.get('avgpool')
model.to(DEVICE)
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vec(arg, model, extraction_layer):
    image = normalize(to_tensor(scaler(arg))).unsqueeze(0).to(DEVICE)
    features = []

    def hook_fn(module, input, output):
        features.append(output.cpu().detach())

    handle = extraction_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image)

    handle.remove()

    if features:
        return features[0].squeeze()
    else:
        return torch.empty(0)

THUMBNAIL_SIZE = (64, 64)

def embed(model, filename: str):
    with Image.open(fp=filename, mode='r') as image:
        return get_vec(arg=image.convert('RGB'), model=model, extraction_layer=extraction_layer).numpy().reshape(OUTPUT_SIZE,)


# https://stackoverflow.com/a/952952
def flatten(arg):
    return [x for xs in arg for x in xs]

def png(filename: str) -> str:
    with Image.open(fp=filename, mode='r') as image:
        buffer = BytesIO()
        # our images are pretty big; let's shrink the hover images to thumbnail size
        image.resize(size=THUMBNAIL_SIZE).convert('RGBA').save(buffer, format='png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def get_picture_from_glob(arg: str, tag: str,) -> list:
    time_get = arrow.now()
    result = [pd.Series(data=[tag, basename(input_file), embed(model=model, filename=input_file), png(filename=input_file)],
                        index=['tag', 'name', 'value', 'png'] )
        for index, input_file in enumerate(tqdm.tqdm(list(iglob(pathname=arg))))]
    print('encoded {} rows of {}  in {}'.format(len(result), tag, arrow.now() - time_get))
    return result


time_start = arrow.now()
train_dict = {basename(folder) : folder + '/*.*' for folder in iglob(TRAIN + '/*') if isdir(folder) }

if not train_dict:
    print(f"Warning: No directories found at {TRAIN}. train_df will be empty. Please ensure the dataset is available at this path.")
    train_df = pd.DataFrame()
else:
    train_df = pd.DataFrame(data=flatten(arg=[get_picture_from_glob(arg=value, tag=key) for key, value in train_dict.items()]))

# Only attempt to modify 'tag' column if train_df is not empty and has the 'tag' column
if not train_df.empty and 'label' in train_df.columns:
    train_df['label'] = train_df['label'].apply(func=lambda x: x.replace(' faces', ''))

print('done in {}'.format(arrow.now() - time_start))

if not train_df.empty and 'label' in train_df.columns:
    print(train_df['filepath'].value_counts(normalize=True).to_frame().T)
else:
    print("train_df is empty or 'label' column is missing. No label counts to display.")

train_reducer = TSNE(random_state=2026, verbose=True, n_jobs=1, perplexity=20.0, init='pca')
train_df[['x', 'y']] = train_reducer.fit_transform(X=train_df['value'].apply(func=pd.Series))

output_notebook()

datasource = ColumnDataSource(train_df[['png', 'tag', 'x', 'y']].sample(n=min(len(train_df) - 1, 10000)))
factor_count = max(train_df['tag'].nunique(), 3)
indices = np.linspace(0, len(Turbo256)-1, factor_count, dtype=int)
palette = [Turbo256[index] for index in indices]
mapper = factor_cmap(field_name = 'tag', palette=palette, factors=train_df['tag'].unique().tolist(), start=0, end=factor_count-1, )

plot_figure = figure(title='TSNE projection: real vs fake faces', width=1000, height=800, tools=('pan, wheel_zoom, reset'))

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@png' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 18px'>@tag</span>
    </div>
</div>
"""))

plot_figure.scatter(x='x', y='y', source=datasource, line_alpha=0.6, fill_alpha=0.6, size=10, color=mapper)
show(plot_figure)

X_train, X_test, y_train, y_test = train_test_split(train_df['value'].apply(pd.Series), train_df['tag'], test_size=0.25, random_state=2026, stratify=train_df['tag'])

logreg = LogisticRegression(max_iter=3000, tol=1e-12).fit(X_train, y_train)
print(f'model fit in {logreg.n_iter_[0]} iterations')
print(f'accuracy: {accuracy_score(y_true=y_test, y_pred=logreg.predict(X=X_test)):5.4f}')
print(f'f1: {f1_score(average='weighted', y_true=y_test, y_pred=logreg.predict(X=X_test)):5.4f}')
print(classification_report(y_true=y_test, y_pred=logreg.predict(X=X_test), zero_division=0.0))