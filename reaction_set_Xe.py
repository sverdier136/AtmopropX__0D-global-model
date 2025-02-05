from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e

species_list = Species([Specie("e", m_e, -e), Specie("Xe", 2.18e-25, 0), Specie("Xe+", 2.18e-25, e)])

Recombinaison ionique

Backscattering:
back_Xe = Reaction(species_list, ["Xe"], ["Xe", "e"], K_reac(), 0 ,[1., 1., 1.])

Ionisation:
ion_Xe = Reaction(sepcies_list, ["Xe"], ["Xe+", "e"], K_iz(T), 1.21, [1., 1., 1.])

Choc élastique:
ela_Xe = ElasticCollisionWithElectron(species_list, ["Xe"], get_K_func(species_list, "N2", "ela_N2"), 0)

Excitation:
exc_Xe = Reaction(species_list, ["Xe"], ["Xe"], K_reac(), 8.32, [1., 1.])

Isotropique:
iso_Xe = Reaction(species_list, ["Xe"], ["Xe"], K_reac(), , )
