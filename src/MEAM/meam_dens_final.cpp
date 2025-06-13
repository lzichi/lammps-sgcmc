// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include "meam.h"
#include "memory.h"

using namespace LAMMPS_NS;

void MEAM::meam_dens_final(int nlocal, int eflag_either, int eflag_global, int eflag_atom,
                           double *eng_vdwl, double *eatom, int /*ntype*/, int *type, int *fmap,
                           double **scale, int &errorflag)
{
  int i, elti;
  int m;
  double rhob, G, dG, Gbar, dGbar, gam, shp[3], Z;
  double denom, rho_bkgd, Fl;
  double scaleii;

  //     Complete the calculation of density

  if (msmeamflag) {
    for (i = 0; i < nlocal; i++) {
      elti = fmap[type[i]];
      if (elti >= 0) {
        scaleii = scale[type[i]][type[i]];
        rho1[i] = 0.0;
        rho2[i] = -1.0 / 3.0 * (arho2b[i] * arho2b[i]
                              - arho2mb[i] * arho2mb[i]);
        rho3[i] = 0.0;
        for (m = 0; m < 3; m++) {
          rho1[i] = rho1[i] + arho1[i][m] * arho1[i][m]
                            - arho1m[i][m] * arho1m[i][m];
          rho3[i] = rho3[i] - 3.0 / 5.0 * (arho3b[i][m] * arho3b[i][m]
                                           - arho3mb[i][m] * arho3mb[i][m]);
        }
        for (m = 0; m < 6; m++) {
          rho2[i] = rho2[i] + v2D[m] * (arho2[i][m] * arho2[i][m]
                                        - arho2m[i][m] * arho2m[i][m]);
        }

        for (m = 0; m < 10; m++) {
          rho3[i] = rho3[i] + v3D[m] * (arho3[i][m] * arho3[i][m]
                                        - arho3m[i][m] * arho3m[i][m]);
        }

        // all the t weights are already accounted for with msmeam
        gamma[i] = rho1[i] + rho2[i] + rho3[i];

        if (rho0[i] > 0.0) {
          gamma[i] = gamma[i] / (rho0[i] * rho0[i]);
        }

        Z = get_Zij(lattce_meam[elti][elti]);

        G = G_gam(gamma[i], ibar_meam[elti], errorflag);
        if (errorflag != 0)
          return;

        get_shpfcn(lattce_meam[elti][elti], stheta_meam[elti][elti], ctheta_meam[elti][elti], shp);

        if (ibar_meam[elti] <= 0) {
          Gbar = 1.0;
          dGbar = 0.0;
        } else {
          if (mix_ref_t == 1) {
            gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
          } else {
            gam = (t1_meam[elti] * shp[0] + t2_meam[elti] * shp[1] + t3_meam[elti] * shp[2]) /
                  (Z * Z);
          }
          Gbar = G_gam(gam, ibar_meam[elti], errorflag);
        }
        rho[i] = rho0[i] * G;

        if (mix_ref_t == 1) {
          if (ibar_meam[elti] <= 0) {
            Gbar = 1.0;
            dGbar = 0.0;
          } else {
            gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
            Gbar = dG_gam(gam, ibar_meam[elti], dGbar);
          }
          rho_bkgd = rho0_meam[elti] * Z * Gbar;
        } else {
          if (bkgd_dyn == 1) {
            rho_bkgd = rho0_meam[elti] * Z;
          } else {
            rho_bkgd = rho_ref_meam[elti];
          }
        }
        rhob = rho[i] / rho_bkgd;
        denom = 1.0 / rho_bkgd;

        G = dG_gam(gamma[i], ibar_meam[elti], dG);

        dgamma1[i] = (G - 2 * dG * gamma[i]) * denom;

        if (!iszero(rho0[i])) {
          dgamma2[i] = (dG / rho0[i]) * denom;
        } else {
          dgamma2[i] = 0.0;
        }

        //     dgamma3 is nonzero only if we are using the "mixed" rule for
        //     computing t in the reference system (which is not correct, but
        //     included for backward compatibility
        if (mix_ref_t == 1) {
          dgamma3[i] = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
        } else {
          dgamma3[i] = 0.0;
        }

        Fl = embedding(A_meam[elti], Ec_meam[elti][elti], rhob, frhop[i]);
        if (eflag_either != 0) {
          Fl *= scaleii;
          if (eflag_global != 0) {
            *eng_vdwl = *eng_vdwl + Fl;
          }
          if (eflag_atom != 0) {
            eatom[i] = eatom[i] + Fl;
          }
        }
      }
    }
  } else {
    for (i = 0; i < nlocal; i++) {
      elti = fmap[type[i]];
      if (elti >= 0) {
        scaleii = scale[type[i]][type[i]];
        rho1[i] = 0.0;
        rho2[i] = -1.0 / 3.0 * arho2b[i] * arho2b[i];
        rho3[i] = 0.0;
        for (m = 0; m < 3; m++) {
          rho1[i] = rho1[i] + arho1[i][m] * arho1[i][m];
          rho3[i] = rho3[i] - 3.0 / 5.0 * arho3b[i][m] * arho3b[i][m];
        }
        for (m = 0; m < 6; m++) {
          rho2[i] = rho2[i] + v2D[m] * arho2[i][m] * arho2[i][m];
        }
        for (m = 0; m < 10; m++) {
          rho3[i] = rho3[i] + v3D[m] * arho3[i][m] * arho3[i][m];
        }

        if (rho0[i] > 0.0) {
          if (ialloy == 1) {
            t_ave[i][0] = fdiv_zero(t_ave[i][0], tsq_ave[i][0]);
            t_ave[i][1] = fdiv_zero(t_ave[i][1], tsq_ave[i][1]);
            t_ave[i][2] = fdiv_zero(t_ave[i][2], tsq_ave[i][2]);
          } else if (ialloy == 2) {
            t_ave[i][0] = t1_meam[elti];
            t_ave[i][1] = t2_meam[elti];
            t_ave[i][2] = t3_meam[elti];
          } else {
            t_ave[i][0] = t_ave[i][0] / rho0[i];
            t_ave[i][1] = t_ave[i][1] / rho0[i];
            t_ave[i][2] = t_ave[i][2] / rho0[i];
          }
        }

        gamma[i] = t_ave[i][0] * rho1[i] + t_ave[i][1] * rho2[i] + t_ave[i][2] * rho3[i];

        if (rho0[i] > 0.0) {
          gamma[i] = gamma[i] / (rho0[i] * rho0[i]);
        }

        Z = get_Zij(lattce_meam[elti][elti]);

        G = G_gam(gamma[i], ibar_meam[elti], errorflag);
        if (errorflag != 0)
          return;

        get_shpfcn(lattce_meam[elti][elti], stheta_meam[elti][elti], ctheta_meam[elti][elti], shp);

        if (ibar_meam[elti] <= 0) {
          Gbar = 1.0;
          dGbar = 0.0;
        } else {
          if (mix_ref_t == 1) {
            gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
          } else {
            gam = (t1_meam[elti] * shp[0] + t2_meam[elti] * shp[1] + t3_meam[elti] * shp[2]) /
                  (Z * Z);
          }
          Gbar = G_gam(gam, ibar_meam[elti], errorflag);
        }
        rho[i] = rho0[i] * G;

        if (mix_ref_t == 1) {
          if (ibar_meam[elti] <= 0) {
            Gbar = 1.0;
            dGbar = 0.0;
          } else {
            gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
            Gbar = dG_gam(gam, ibar_meam[elti], dGbar);
          }
          rho_bkgd = rho0_meam[elti] * Z * Gbar;
        } else {
          if (bkgd_dyn == 1) {
            rho_bkgd = rho0_meam[elti] * Z;
          } else {
            rho_bkgd = rho_ref_meam[elti];
          }
        }
        rhob = rho[i] / rho_bkgd;
        denom = 1.0 / rho_bkgd;

        G = dG_gam(gamma[i], ibar_meam[elti], dG);

        dgamma1[i] = (G - 2 * dG * gamma[i]) * denom;

        if (!iszero(rho0[i])) {
          dgamma2[i] = (dG / rho0[i]) * denom;
        } else {
          dgamma2[i] = 0.0;
        }

        //     dgamma3 is nonzero only if we are using the "mixed" rule for
        //     computing t in the reference system (which is not correct, but
        //     included for backward compatibility
        if (mix_ref_t == 1) {
          dgamma3[i] = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
        } else {
          dgamma3[i] = 0.0;
        }

        Fl = embedding(A_meam[elti], Ec_meam[elti][elti], rhob, frhop[i]);

        if (eflag_either != 0) {
          Fl *= scaleii;
          if (eflag_global != 0) {
            *eng_vdwl = *eng_vdwl + Fl;
          }
          if (eflag_atom != 0) {
            eatom[i] = eatom[i] + Fl;

          }
        }
      }
    }
  }
}

void MEAM::meam_dens_final_one_atom(int i, double *eng_vdwl, double *eatom_local, int itype, 
                                    int ifmap, double *iscale, int &errorflag) 
{

  int elti;
  int m;
  double rhob, G, dG, Gbar, dGbar, gam, shp[3], Z;
  double denom, rho_bkgd, Fl;
  double scaleii;

  // double irho1, irho2, irho3, igamma, idgamma1, idgamma2, idgamma3;
  // double *it_ave;
  // int size = 3;
  // memory->create(it_ave, size, "pair::it_ave");

  // complete density calculation

  // TODO: msmeam support
  
  // elti = ifmap;
  // if (elti >= 0) {
  //   scaleii = iscale[itype];
  //   rho1_eng[i] = 0.0;
  //   rho2_eng[i] = -1.0 / 3.0 * arho2b[i] * arho2b[i];
  //   rho3_eng[i] = 0.0;
  //   for (m = 0; m < 3; m++) {
  //     rho1_eng[i] = rho1[i] + arho1[i][m] * arho1[i][m];
  //     rho3_eng[i] = rho3[i] - 3.0 / 5.0 * arho3b[i][m] * arho3b[i][m];
  //   }
  //   for (m = 0; m < 6; m++) {
  //     rho2_eng[i] = rho2[i] + v2D[m] * arho2[i][m] * arho2[i][m];
  //   }
  //   for (m = 0; m < 10; m++) {
  //     rho3_eng[i] = rho3[i] + v3D[m] * arho3[i][m] * arho3[i][m];
  //   }

  //   if (rho0[i] > 0.0) {
  //     if (ialloy == 1) {
  //       t_ave_eng[0] = fdiv_zero(t_ave[i][0], tsq_ave[i][0]);
  //       t_ave_eng[1] = fdiv_zero(t_ave[i][1], tsq_ave[i][1]);
  //       t_ave_eng[2] = fdiv_zero(t_ave[i][2], tsq_ave[i][2]);
  //     } else if (ialloy == 2) {
  //       t_ave_eng[0] = t1_meam[elti];
  //       t_ave_eng[1] = t2_meam[elti];
  //       t_ave_eng[2] = t3_meam[elti];
  //     } else {
  //       t_ave_eng[0] = t_ave_eng[0] / rho0[i];
  //       t_ave_eng[1] = t_ave_eng[1] / rho0[i];
  //       t_ave_eng[2] = t_ave_eng[2] / rho0[i];
  //     }
  //   }

  //   igamma = t_ave_eng[0] * rho1_eng[i] + t_ave_eng[1] * rho2_eng[i] + t_ave_eng[2] * rho3_eng[i];

  //   if (rho0[i] > 0.0) {
  //     igamma = igamma / (rho0[i] * rho0[i]);
  //   }

  //   Z = get_Zij(lattce_meam[elti][elti]);

  //   G = G_gam(gamma[i], ibar_meam[elti], errorflag);
  //   if (errorflag != 0)
  //     return;

  //   get_shpfcn(lattce_meam[elti][elti], stheta_meam[elti][elti], ctheta_meam[elti][elti], shp);

  //   if (ibar_meam[elti] <= 0) {
  //     Gbar = 1.0;
  //     dGbar = 0.0;
  //   } else {
  //     if (mix_ref_t == 1) {
  //       gam = (t_ave_eng[0] * shp[0] + t_ave_eng[1] * shp[1] + t_ave_eng[2] * shp[2]) / (Z * Z);
  //     } else {
  //       gam = (t1_meam[elti] * shp[0] + t2_meam[elti] * shp[1] + t3_meam[elti] * shp[2]) /
  //             (Z * Z);
  //     }
  //     Gbar = G_gam(gam, ibar_meam[elti], errorflag);
  //   }
  //   rho[i] = rho0[i] * G;

  //   if (mix_ref_t == 1) {
  //     if (ibar_meam[elti] <= 0) {
  //       Gbar = 1.0;
  //       dGbar = 0.0;
  //     } else {
  //       gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
  //       Gbar = dG_gam(gam, ibar_meam[elti], dGbar);
  //     }
  //     rho_bkgd = rho0_meam[elti] * Z * Gbar;
  //   } else {
  //     if (bkgd_dyn == 1) {
  //       rho_bkgd = rho0_meam[elti] * Z;
  //     } else {
  //       rho_bkgd = rho_ref_meam[elti];
  //     }
  //   }
  //   rhob = rho[i] / rho_bkgd;
  //   denom = 1.0 / rho_bkgd;

  //   G = dG_gam(gamma[i], ibar_meam[elti], dG);

  //   idgamma1 = (G - 2 * dG * gamma[i]) * denom;

  //   if (!iszero(rho0[i])) {
  //     idgamma2 = (dG / rho0[i]) * denom;
  //   } else {
  //     idgamma2 = 0.0;
  //   }

  //   //     dgamma3 is nonzero only if we are using the "mixed" rule for
  //   //     computing t in the reference system (which is not correct, but
  //   //     included for backward compatibility
  //   if (mix_ref_t == 1) {
  //     idgamma3 = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
  //   } else {
  //     idgamma3 = 0.0;
  //   }

  //   Fl = embedding(A_meam[elti], Ec_meam[elti][elti], rhob, frhop[i]);

  //   Fl *= scaleii;
  //  // *Ei =  *Ei + Fl;
  //   eatom_local[i] = Fl;
  // }

  // memory->destroy(it_ave);


    elti = ifmap;
    if (elti >= 0) {
      scaleii = iscale[itype];
      rho1[i] = 0.0;
      rho2[i] = -1.0 / 3.0 * arho2b[i] * arho2b[i];
      rho3[i] = 0.0;
      for (m = 0; m < 3; m++) {
        rho1[i] = rho1[i] + arho1[i][m] * arho1[i][m];
        rho3[i] = rho3[i] - 3.0 / 5.0 * arho3b[i][m] * arho3b[i][m];
      }
      for (m = 0; m < 6; m++) {
        rho2[i] = rho2[i] + v2D[m] * arho2[i][m] * arho2[i][m];
      }
      for (m = 0; m < 10; m++) {
        rho3[i] = rho3[i] + v3D[m] * arho3[i][m] * arho3[i][m];
      }

      if (rho0[i] > 0.0) {
        if (ialloy == 1) {
          t_ave[i][0] = fdiv_zero(t_ave[i][0], tsq_ave[i][0]);
          t_ave[i][1] = fdiv_zero(t_ave[i][1], tsq_ave[i][1]);
          t_ave[i][2] = fdiv_zero(t_ave[i][2], tsq_ave[i][2]);
        } else if (ialloy == 2) {
          t_ave[i][0] = t1_meam[elti];
          t_ave[i][1] = t2_meam[elti];
          t_ave[i][2] = t3_meam[elti];
        } else {
          t_ave[i][0] = t_ave[i][0] / rho0[i];
          t_ave[i][1] = t_ave[i][1] / rho0[i];
          t_ave[i][2] = t_ave[i][2] / rho0[i];
        }
      }

      gamma[i] = t_ave[i][0] * rho1[i] + t_ave[i][1] * rho2[i] + t_ave[i][2] * rho3[i];

      if (rho0[i] > 0.0) {
        gamma[i] = gamma[i] / (rho0[i] * rho0[i]);
      }

      Z = get_Zij(lattce_meam[elti][elti]);

      G = G_gam(gamma[i], ibar_meam[elti], errorflag);
      if (errorflag != 0)
        return;

      get_shpfcn(lattce_meam[elti][elti], stheta_meam[elti][elti], ctheta_meam[elti][elti], shp);

      if (ibar_meam[elti] <= 0) {
        Gbar = 1.0;
        dGbar = 0.0;
      } else {
        if (mix_ref_t == 1) {
          gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
        } else {
          gam = (t1_meam[elti] * shp[0] + t2_meam[elti] * shp[1] + t3_meam[elti] * shp[2]) /
                (Z * Z);
        }
        Gbar = G_gam(gam, ibar_meam[elti], errorflag);
      }
      rho[i] = rho0[i] * G;

      if (mix_ref_t == 1) {
        if (ibar_meam[elti] <= 0) {
          Gbar = 1.0;
          dGbar = 0.0;
        } else {
          gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
          Gbar = dG_gam(gam, ibar_meam[elti], dGbar);
        }
        rho_bkgd = rho0_meam[elti] * Z * Gbar;
      } else {
        if (bkgd_dyn == 1) {
          rho_bkgd = rho0_meam[elti] * Z;
        } else {
          rho_bkgd = rho_ref_meam[elti];
        }
      }
      rhob = rho[i] / rho_bkgd;
      denom = 1.0 / rho_bkgd;

      G = dG_gam(gamma[i], ibar_meam[elti], dG);

      dgamma1[i] = (G - 2 * dG * gamma[i]) * denom;

      if (!iszero(rho0[i])) {
        dgamma2[i] = (dG / rho0[i]) * denom;
      } else {
        dgamma2[i] = 0.0;
      }

      //     dgamma3 is nonzero only if we are using the "mixed" rule for
      //     computing t in the reference system (which is not correct, but
      //     included for backward compatibility
      if (mix_ref_t == 1) {
        dgamma3[i] = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
      } else {
        dgamma3[i] = 0.0;
      }

      Fl = embedding(A_meam[elti], Ec_meam[elti][elti], rhob, frhop[i]);

      Fl *= scaleii;
      eatom_local[i] = eatom_local[i] + Fl;
      // printf("eatom_local[i], %g, i %d \n", eatom_local[i], i);

    
  }
}


