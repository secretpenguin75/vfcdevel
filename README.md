coucou
coucou EMMA
alut adrien 
ça marche
sur blocnote
# Quickstart
Dans un dossier de travail faire

```
git clone git@github.com:secretpenguin75/vfcdevel.git
```

Ensuite creer un fichier jupyter notebook et importer la fonction carotte avec

```
from vfcdevel.profilegen import Profile_gen
```

Et ensuite, appeler les fonctions

```
depth,depthwe,rho,iso,iso_diff,date,sigma = Profile_gen(df['year'],df['tp'],df['d18O_inter'],Tmean,350)

```

```
ICORDA = pd.read_excel("./data_emma/ICORDA_depth_age_d18O_model_period.xlsx")
df = pd.read_csv('./vfcdevel/data/lmdz_DC.csv',index_col=0,parse_dates = True)

from vfcdevel.runvfc import superplot
superplot(df,ICORDA)

```
etc...

