
#include "header.h"


static const int32_t num_class[] = {  1, };

int32_t get_num_target(void) {
  return N_TARGET;
}
void get_num_class(int32_t* out) {
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t get_num_feature(void) {
  return 5;
}
const char* get_threshold_type(void) {
  return "float32";
}
const char* get_leaf_output_type(void) {
  return "float32";
}

void predict(union Entry* data, int pred_margin, float* result) {
  unsigned int tmp;
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.0021553779952)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.083231970668)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.16852813959)) {
        result[0] += 0.48228878;
      } else {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.052120149136)) {
          result[0] += 0.46440205;
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.021610029042)) {
            result[0] += -0.5501071;
          } else {
            result[0] += 0.43696955;
          }
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.063753969967)) {
        result[0] += 0.44750592;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.053529631346)) {
          result[0] += -0.65146685;
        } else {
          result[0] += 0.37670347;
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.62224674225)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.072661869228)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.11715260893)) {
          result[0] += 0.47522902;
        } else {
          if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.36554917693)) {
            result[0] += 0.33621812;
          } else {
            result[0] += -0.45590702;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.041482325643)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.047299910337)) {
            result[0] += -0.090008914;
          } else {
            result[0] += -0.78471255;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.014470786788)) {
            result[0] += 0.003624403;
          } else {
            result[0] += 0.4720408;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.96456557512)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.14766092598)) {
          result[0] += 0.44081628;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.046446308494)) {
            result[0] += -0.7801283;
          } else {
            result[0] += 0.43235824;
          }
        }
      } else {
        result[0] += -0.7805863;
      }
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.0021553779952)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.10456574708)) {
      result[0] += 0.41025075;
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.063753969967)) {
        result[0] += 0.3978756;
      } else {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.023819100112)) {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.052052356303)) {
            result[0] += -0.47984233;
          } else {
            result[0] += 0.21716517;
          }
        } else {
          result[0] += 0.34538195;
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.47249862552)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.066851854324)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.11715260893)) {
          result[0] += 0.40398514;
        } else {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.53274011612)) {
            result[0] += 0.40875232;
          } else {
            result[0] += -0.3473879;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.049052976072)) {
          result[0] += -0.5180066;
        } else {
          result[0] += 0.4032946;
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0501489639)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.13277551532)) {
          result[0] += 0.40633845;
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.15108849108)) {
            result[0] += -0.5195771;
          } else {
            result[0] += 0.41220444;
          }
        }
      } else {
        result[0] += -0.5223684;
      }
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.0021553779952)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.10456574708)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.16852813959)) {
        result[0] += 0.37374365;
      } else {
        if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.83600616455)) {
          if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.14721438289)) {
            result[0] += 0.124694236;
          } else {
            result[0] += 0.34791407;
          }
        } else {
          result[0] += -0.11473224;
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.063753969967)) {
        result[0] += 0.35763454;
      } else {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.023819100112)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.033004757017)) {
            result[0] += 0.12670204;
          } else {
            result[0] += -0.38164678;
          }
        } else {
          result[0] += 0.3060323;
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.47249862552)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.066851854324)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.11715260893)) {
          result[0] += 0.36492598;
        } else {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.53274011612)) {
            result[0] += 0.35888034;
          } else {
            result[0] += -0.25384542;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.049052976072)) {
          result[0] += -0.42777595;
        } else {
          result[0] += 0.3630625;
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0501489639)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.13277551532)) {
          result[0] += 0.36407188;
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.15108849108)) {
            result[0] += -0.425859;
          } else {
            result[0] += 0.3655861;
          }
        }
      } else {
        result[0] += -0.4308218;
      }
    }
  }
  
  // Apply base_scores
  result[0] += 0.4922882798795578663;
  
  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void postprocess(float* result) {
  // sigmoid
  const float alpha = (float)1;
  for (size_t i = 0; i < N_TARGET * MAX_N_CLASS; ++i) {
    result[i] = (float)(1) / ((float)(1) + expf(-alpha * result[i]));
  }
}

