from activ.ct.convert import main


def test_convert():
    main(['data/ct/115Label_fake', 'ct_convert.csv'])
