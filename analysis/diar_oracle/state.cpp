// Validation adapter over unchanged upstream AOSC; no inference backend needed.
#include "aosc_state.h"
using namespace nemo_speech::asr;
extern "C" {
void* state_new(const float* silence) {
 DiarGeometry geo{264,40,340,300,0,40};DiarScoringConfig scoring;scoring.sil_frames_per_spk=1;
 return new AoscState(geo,scoring,8,512,std::vector<float>(silence,silence+512));
}
void state_update(void* p,const float* emb,int t,const float* probs,int lc,int rc) {
 ((AoscState*)p)->update(emb,t,probs,lc,rc);
}
int state_cache_frames(void* p){return ((AoscState*)p)->spkcache_frames();}
int state_fifo_frames(void* p){return ((AoscState*)p)->fifo_frames();}
const float* state_cache(void* p){return ((AoscState*)p)->spkcache().data();}
const float* state_fifo(void* p){return ((AoscState*)p)->fifo().data();}
const float* state_probs(void* p){return ((AoscState*)p)->spkcache_preds().data();}
void state_delete(void* p){delete (AoscState*)p;}
}
