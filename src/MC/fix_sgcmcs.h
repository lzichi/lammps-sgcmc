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

/* ----------------------------------------------------------------------
 * Parallel Monte-Carlo code for the semi-grandcanonical ensemble (SGC)
 * and the variance-constrained semi-grandcanonical ensemble (VC-SGC).
 *
 * See Sadigh et al., Phys. Rev. B 85, 184203 (2012) for a
 * description of the algorithm.
 *
 * Code author: Alexander Stukowski (stukowski@mm.tu-darmstadt.de)
 *
 * Updates to use the sectoring method for parallelism: Aidan Thompson, SNL,
 * Laura Zichi, SNL and Axel Kohlmeyer, Temple U
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(sgcmcs, FixSemiGrandCanonicalMCSector);
// clang-format on
#else

#ifndef FIX_SGCMCS_H
#define FIX_SGCMCS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSemiGrandCanonicalMCSector : public Fix {
 public:
  FixSemiGrandCanonicalMCSector(class LAMMPS *, int, char **);

  ~FixSemiGrandCanonicalMCSector() override;

  int setmask() override;
  void init() override;
  void init_list(int id, class NeighList *ptr) override;
  void post_force(int vflag) override;
  double compute_vector(int index) override;

  double memory_usage() override;

  int pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) override;
  void unpack_forward_comm(int n, int first, double *buf) override;
//   int pack_reverse_comm(int n, int first, double *buf) override;
//   void unpack_reverse_comm(int n, int *list, double *buf) override;

  /******************** Monte-Carlo routines ************************/

  // This routine does one full MC step.
  void doMC();

  // Calculates the change in energy that swapping the given atom would produce.
  // This routine is for the general case of an arbitrary potential and
  // IS VERY SLOW! It computes the total energies of the system for the unmodified state
  // and for the modified state and then returns the difference of both values.
  // This routine should only be used for debugging purposes.
  double computeEnergyChangeGeneric(int flipAtom, int oldSpecies, int newSpecies);

  // Calculates the change in energy that swapping the given atom would produce.
  // This uses the atomic energy method
  double computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies);

  // Lets LAMMPS calculate the total potential energy of the system.
  double computeTotalEnergy();

  // Flips the type of one atom.
  // This routine is for the generic case.
  void flipAtomGeneric(int flipAtom, int oldSpecies, int newSpecies);

  // Flips the type of one atom.
  // This routine is for the atomic energy method
  void flipAtomEatom(int flipAtom, int oldSpecies, int newSpecies);

  // Transfers the locally atom types to the neighbors.
  void communicateTypes();

  /*********** Sectoring method for parallelism routines ************/

  // Set up sectoring on each processor
  void sectoring(); 

  // Map atom coordinates to assigned sector
  int coords2sector(double *);

  void setup_post_neighbor() override;

  void post_neighbor() override;

 private:
  // The number of MD steps between each MC cycle.
  int nevery_mdsteps;

  // The fraction of atoms that should be swapped per MC step.
  double swap_fraction;

  // The maximum interaction radius of all potentials.
  double interactionRadius;

  // The inverse MC temperature.
  double beta;

  // Chemical potential differences for all species. The differences are relative to the chemical
  // potential of the first species. Note that this array is based on index 1 (not 0 as normal C arrays).
  // This means the first two elements of this vector are always zero.
  std::vector<double> deltamu;

  // The MC variance constraint parameter.
  double kappa;

  // The target concentration values for each species. The concentration of first species is
  // implicitely defined as one minues all other concentrations. Please note that this vector
  // is based on index 1. The first element at index 0 is not used.
  std::vector<double> targetConcentration;

  // The master seed value for the random number generators on all nodes.
  int seed;

  // The random number generator that is in sync with all other nodes.
  class RanPark *random;

  // The local random number generator for this proc only.
  class RanPark *localRandom;

  // The total number of atoms of the different species in the whole system.
  // Divide this by the total number of atoms to get the global concentration.
  // Since LAMMPS atom types start at index 1 this array is also based on index 1.
  // The first array element at index 0 is not used.
  std::vector<int> speciesCounts;

  // The full neighbor list used by this fix.
  class NeighList *neighborList;

  // This counter indicates the current MPI communication stage to let the
  // pack/unpack routines know which data is being transmitted.
  int communicationStage;

  // The total number of accepted swaps during the last MC step.
  int nAcceptedSwaps;

  // The total number of rejected swaps during the last MC step.
  int nRejectedSwaps;

  // A compute used to compute the total potential energy of the system.
  class Compute *compute_pe;

  // Indicate whether or not atomic energy method is used
  int atomicenergyflag;

  /*********** Sectoring method for parallelism variables ************/

  // sector_flag = 0 if serial algorithm
  // sector_flag = 1 if parallel algorithm
  int sector_flag; 

  // max value of nlocal (for size of lists)
  int nlocal_max; 

  int nsectors;
  double *rsec;

  // stacking variables for sectoring algorithm

  // number of atoms in each sector
  int *num_atoms_per_sector;

  // continuous list of atom ids per sector 
  int *atoms_in_sector;

  // number of atoms a processor owns
  int numFixAtomsLocal;
            
};
}

#endif
#endif
