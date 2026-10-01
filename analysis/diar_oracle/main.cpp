// Validation-only executable: invokes unchanged upstream CPU model/state.
#include "diar_pipeline.h"
#include <fstream>
#include <iostream>
#include <chrono>
using namespace nemo_speech::asr;
int main(int argc,char** argv) {
 if(argc<4) { std::cerr<<"model.gguf audio.wav output-prefix [full|stream|chunk]\n"; return 2; }
 ggml_runtime::Params params;params.use_gpu=false;
 ggml_runtime::BackendManager backend(params);
 DiarModel model(backend,argv[1]);
 std::vector<float> audio; int rate;
 if(argc<5 || std::string(argv[4])!="chunk") {
  if(!read_wav_mono_16k(argv[2],audio,rate))return 3;
 }
 const std::string mode=argc>4?argv[4]:"stream";
 std::vector<float> probs;
 auto start=std::chrono::steady_clock::now();
 if(mode=="full") {
  int64_t frames;probs=model.diarize_offline(audio.data(),audio.size(),&frames);
 } else if(mode=="chunk") {
  // A raw mel file allows complete-model comparisons independent of FFT.
  std::ifstream f(argv[2],std::ios::binary|std::ios::ate);
  size_t n=f.tellg();f.seekg(0);std::vector<float> mel(n/4);f.read((char*)mel.data(),n);
  auto out=model.model().run_chunk(mel.data(),mel.size()/128,nullptr,0,nullptr,0);
  probs=out.native_preds;
  std::ofstream e(std::string(argv[3])+".emb",std::ios::binary);e.write((char*)out.chunk_embs.data(),out.chunk_embs.size()*4);
 } else {
  DiarGeometry geo{264,40,340,300,0,40};
  DiarStream stream(model,geo);
  // Keep reference frontend/state bounded while feeding 1-second slices.
  for(size_t s=0;s<audio.size();s+=16000)stream.feed_audio(audio.data()+s,std::min<size_t>(16000,audio.size()-s));
  stream.finish();probs=stream.frame_probs();
 }
 std::ofstream f(std::string(argv[3])+".probs",std::ios::binary);f.write((char*)probs.data(),probs.size()*4);
 std::cout<<"frames="<<probs.size()/8<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
}
