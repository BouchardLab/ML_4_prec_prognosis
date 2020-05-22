from activ.clustering.pbsim import main


def test_pbsim():
    main(['--no_residuals', 'pbsim_data.h5'])
