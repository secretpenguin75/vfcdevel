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


nouveau package : pip install varname

Modif push 25/07 :

> ajout mixing_scale et noise_scale dans inputs fonction superplot. Mon local.py ressemble à ça:
df.precipd18O-=4.1
noise_scale_mm=10 ; mixing_scale_mm=40   #à modif
superplot(df,ICORDA, 'ICORDA', noise_scale_mm, mixing_scale_mm)

> Nouvelle fonction core_details dans runvfc.py avec la résolution des obs (ICORDA ou Subglacior), les couleurs, la grille régulière pour préparer les spectres.
Les listes core resolution correspondent au détail de la résolution en discret sur les premiers mètres de la carotte qui n'est pas homogène.
Si on ne veut pas utiliser la résolution des OBS il reste la ligne 
#df_int = block_average(df,.01) # block average at 1cm resolution  
commentée dans profilegen_v2
La maille grille régulière (grid) corespond à la valeur moyenne des résolutions en discret.

> Merge Profile_gen_legacy et Profile_gen

> Fichier plot.py renommé make_pretty_figures.py (tu peux rechanger mais plot c'était confus pour moi)

> Plt.step plutôt que ma fonction plot_stairsteps

> lundi j'ajoute la storage diffusion pour Subglacior

> Prêt pour ICORDA et (quasi) Subglacior. Enjoy :)