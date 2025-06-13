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
   Contributing author: Laura Zichi (SNL)
------------------------------------------------------------------------- */

#include "compute_localenergy_atom.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "comm.h"

#include <cstring>

using namespace LAMMPS_NS;

/*********************************************************************
* Constructs the compute object and parses the input parameters
*********************************************************************/

ComputeLocalenergyAtom::ComputeLocalenergyAtom(LAMMPS *lmp, int narg, char **arg) :
Compute(lmp, narg, arg), energy(nullptr), list(nullptr)
{
   if (narg != 3) error->all(FLERR, "Illegal compute localenergy/atom command");

   peratom_flag = 1; // compute per atom local energy
   size_peratom_cols = 0;

   // check that pair style supports compute_atomic_energy
   if (!force->pair->atomic_energy_enable) {
      error->all(FLERR, "compute localenergy/atom - Pair style does not support atomic energy method");
   }

   printf("Inside compute localenergy!");

   nmax = 0;
   comm_reverse = 1;
}

/*********************************************************************
* Clean up list of calculated local energies
*********************************************************************/

ComputeLocalenergyAtom::~ComputeLocalenergyAtom()
{
   memory->destroy(energy);
}

/*********************************************************************
* Initalize compute by getting full neighbor list and ghost atoms
*********************************************************************/

void ComputeLocalenergyAtom::init() {

   // atomic energy method needs ghost neighbors
   auto req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

}

/*********************************************************************
* Assigns the requested neighbor list to the compute
*********************************************************************/

void ComputeLocalenergyAtom::init_list(int /*id*/, NeighList *ptr) 
{
   list = ptr;
}

/*********************************************************************
* Calculate local atomic energy for each atom
*********************************************************************/

void ComputeLocalenergyAtom::compute_peratom()
{

   int i;
   double Ei = 0;

   invoked_peratom = update->ntimestep;

   // grow Ei array if necessary

   if (atom->nmax > nmax) {
      memory->destroy(energy);
      nmax = atom->nmax;
      memory->create(energy, nmax, "localenergy/atom:energy");
      vector_atom = energy;
   }

   // invoke full neighbor list (will copy or build if necessary)

   int inum = list->inum;
   int *ilist = list->ilist;

   // get per-atom local energy

   for (int ii = 0; ii < atom->nlocal + atom->nghost; ii++) {

      Ei = force->pair->compute_atomic_energy(ii, list); 
      energy[ii] = Ei;
   }

   comm->reverse_comm(this);

   int *mask = atom->mask;

  for (i = 0; i < atom->nlocal; i++)
    if (!(mask[i] & groupbit)) energy[i] = 0.0;
}

/*********************************************************************
* Memory usage of local atom-based array
*********************************************************************/

double ComputeLocalenergyAtom::memory_usage()
{
  double bytes = (double) nmax * sizeof(double);
  return bytes;
}

int ComputeLocalenergyAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
   
   buf[m++] = energy[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeLocalenergyAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    energy[j] += buf[m++];
  }
}
