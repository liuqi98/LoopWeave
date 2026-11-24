from train.continue_peptide_geometry_380_v1 import structure_score, geometry_guards, pareto_epochs


def metrics():
    return dict(aligned_rmsd=2.,rmsd=3.,interface_fape=.7,contact_f1=.5,
                cn_mae=.3,omega_planarity_deviation_deg=20.,clash=.001,aar=.4)


def test_selection_ignores_aar_and_prefers_structure():
    b=metrics();m={**b,'aar':0.,'aligned_rmsd':1.9,'contact_f1':.6}
    assert structure_score(m,b)<structure_score(b,b)
    assert all(geometry_guards(m,b).values())


def test_clash_and_geometry_guards():
    b=metrics()
    assert not geometry_guards({**b,'clash':.002},b)['clash']
    assert not geometry_guards({**b,'cn_mae':.31},b)['cn']


def test_pareto_retains_tradeoffs():
    b=metrics();better={**b,'aligned_rmsd':1.8};trade={**b,'aligned_rmsd':2.1,'contact_f1':.7}
    rows=[dict(epoch=i,validation=dict(summary=m)) for i,m in enumerate([b,better,trade])]
    assert pareto_epochs(rows)==[1,2]
