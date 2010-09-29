#ifndef _AUX_FUNCTIONS_INCLUDED_
#define _AUX_FUNCTIONS_INCLUDED_

#include <vector>

namespace onbnn
{
  /*
   * Function to make the GLPK optimisation less verbose.
   */
  static int glp_hook( void* info, const char* s )
  {
    return 1;
  }
}

#endif
