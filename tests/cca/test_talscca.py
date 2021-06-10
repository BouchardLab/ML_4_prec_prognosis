from activ.cca.alscca import TALSCCA
from activ.readfile import load_data
tbifile = load_data()
talscca = TALSCCA(scale=True)
talscca.fit(tbifile.biomarkers, tbifile.outcomes)
bm_cv, oc_cv = talscca.transform(tbifile.biomarkers, tbifile.outcomes)

talscca = TALSCCA(scale=True, n_components=3)
talscca.fit(tbifile.biomarkers, tbifile.outcomes)
bm_cv, oc_cv = talscca.transform(tbifile.biomarkers, tbifile.outcomes)
