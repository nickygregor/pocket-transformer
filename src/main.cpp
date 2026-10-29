// Pocket Transformer: one-block, byte-level causal transformer inference.
// No ML runtime dependencies. Check model.py for the binary format/training.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Vec=std::vector<float>;
uint32_t u32(std::istream& in){unsigned char b[4];if(!in.read(reinterpret_cast<char*>(b),4))throw std::runtime_error("Truncated model");return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
float f32(std::istream& in){uint32_t u=u32(in);float f;std::memcpy(&f,&u,4);if(!std::isfinite(f))throw std::runtime_error("Non-finite model weight");return f;}
struct Matrix{
    int rows,cols;bool quantized;Vec weights,scales;std::vector<int8_t> bytes;
    Matrix(std::istream& in,int r,int c,bool q):rows(r),cols(c),quantized(q){
        if(q){scales.resize(r);for(float& s:scales){s=f32(in);if(s<=0)throw std::runtime_error("Invalid quantization scale");}bytes.resize(size_t(r)*c);if(!in.read(reinterpret_cast<char*>(bytes.data()),bytes.size()))throw std::runtime_error("Truncated int8 weights");}
        else {weights.resize(size_t(r)*c);for(float& w:weights)w=f32(in);}
    }
    float at(int r,int c)const{return quantized?float(bytes[size_t(r)*cols+c])*scales[r]:weights[size_t(r)*cols+c];}
    Vec row(int r)const{Vec v(cols);for(int c=0;c<cols;c++)v[c]=at(r,c);return v;}
    Vec mul(const Vec& x)const{Vec out(rows,0);for(int r=0;r<rows;r++){float sum=0;if(quantized){for(int c=0;c<cols;c++)sum+=float(bytes[size_t(r)*cols+c])*x[c];sum*=scales[r];}else for(int c=0;c<cols;c++)sum+=weights[size_t(r)*cols+c]*x[c];out[r]=sum;}return out;}
    size_t storage()const{return weights.size()*4+scales.size()*4+bytes.size();}
};
Vec norm(const Vec& x){float ss=0;for(float v:x)ss+=v*v;float inverse=1/std::sqrt(ss/x.size()+1e-5f);Vec y=x;for(float& v:y)v*=inverse;return y;}
void add(Vec& a,const Vec& b){for(size_t i=0;i<a.size();i++)a[i]+=b[i];}
struct Model{
    int d,h,f,context,vocab;bool quantized;std::vector<Matrix> m;
    explicit Model(const std::string& file){
        std::ifstream in(file,std::ios::binary);if(!in)throw std::runtime_error("Cannot open model");char magic[8];in.read(magic,8);if(in.gcount()!=8||std::string(magic,8)!="PTLM0001")throw std::runtime_error("Invalid model magic");
        d=int(u32(in));h=int(u32(in));f=int(u32(in));context=int(u32(in));vocab=int(u32(in));uint32_t q=u32(in);quantized=q==1;
        if(d<4||d>512||h<1||h>16||d%h||f<4||f>2048||context<2||context>2048||vocab!=256||q>1)throw std::runtime_error("Invalid or unsupported model dimensions");
        for(auto shape:std::vector<std::pair<int,int>>{{vocab,d},{context,d},{d,d},{d,d},{d,d},{d,d},{f,d},{d,f},{vocab,d}})m.emplace_back(in,shape.first,shape.second,quantized);
        char extra;if(in.get(extra))throw std::runtime_error("Unexpected trailing model data");
    }
    size_t bytes()const{size_t result=0;for(auto& a:m)result+=a.storage();return result;}
};
struct Session{
    const Model& model;std::vector<Vec> keys,values;
    explicit Session(const Model& m):model(m){}
    Vec step(int token){
        const int pos=int(keys.size()),d=model.d,hd=d/model.h;
        if(token<0||token>=model.vocab||pos>=model.context)throw std::runtime_error("Token or context out of bounds");
        auto& m=model.m;Vec x=m[0].row(token);add(x,m[1].row(pos));Vec z=norm(x);
        Vec q=m[2].mul(z);keys.push_back(m[3].mul(z));values.push_back(m[4].mul(z));Vec attention(d,0);
        for(int head=0;head<model.h;head++){
            Vec scores(pos+1,0);float maxscore=-std::numeric_limits<float>::infinity();
            for(int t=0;t<=pos;t++){for(int j=0;j<hd;j++)scores[t]+=q[head*hd+j]*keys[t][head*hd+j];scores[t]/=std::sqrt(float(hd));maxscore=std::max(maxscore,scores[t]);}
            float total=0;for(float& s:scores){s=std::exp(s-maxscore);total+=s;}
            for(int t=0;t<=pos;t++)for(int j=0;j<hd;j++)attention[head*hd+j]+=scores[t]/total*values[t][head*hd+j];
        }
        add(x,m[5].mul(attention));Vec hidden=m[6].mul(norm(x));for(float& a:hidden)a=std::max(0.f,a);add(x,m[7].mul(hidden));return m[8].mul(norm(x));
    }
};
std::string escaped(const std::string& s){std::string out;const char* hex="0123456789abcdef";for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c>=32&&c<127)out+=char(c);else{out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}}return out;}
int main(int argc,char** argv){try{
    std::string file,prompt="the ";int steps=64;unsigned seed=7;float temperature=0;bool cached=true,logitsOnly=false; 
    for(int i=1;i<argc;i++){std::string arg=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("Missing argument value");return std::string(argv[i]);};
        if(arg=="--model")file=value();else if(arg=="--prompt")prompt=value();else if(arg=="--steps")steps=std::stoi(value());else if(arg=="--temperature")temperature=std::stof(value());else if(arg=="--seed")seed=unsigned(std::stoul(value()));else if(arg=="--no-cache")cached=false;else if(arg=="--logits")logitsOnly=true;else throw std::runtime_error("Unknown option: "+arg);
    }
    if(file.empty()||prompt.empty()||steps<0||!std::isfinite(temperature)||temperature<0)throw std::runtime_error("Use --model FILE --prompt TEXT [--steps N] [--temperature T] [--no-cache] [--logits]");
    Model model(file);if(prompt.size()>size_t(model.context)||size_t(steps)>size_t(model.context)-prompt.size())throw std::runtime_error("Prompt + steps exceed model context");
    std::vector<int> tokens;for(unsigned char c:prompt)tokens.push_back(c);Session session(model);Vec logits;
    auto begin=std::chrono::steady_clock::now();for(int t:tokens)logits=session.step(t);
    if(logitsOnly){std::cout<<std::setprecision(9)<<"[";for(size_t i=0;i<logits.size();i++)std::cout<<(i?",":"")<<logits[i];std::cout<<"]\n";return 0;}
    auto prefill=std::chrono::steady_clock::now();std::mt19937 rng(seed);std::string generated;
    for(int n=0;n<steps;n++){
        int token;if(temperature==0)token=int(std::max_element(logits.begin(),logits.end())-logits.begin());else{Vec p=logits;float max=*std::max_element(p.begin(),p.end());for(float& x:p)x=std::exp((x-max)/temperature);std::discrete_distribution<int> choose(p.begin(),p.end());token=choose(rng);}
        generated+=char(token);tokens.push_back(token);
        if(n+1<steps){if(cached)logits=session.step(token);else{Session full(model);for(int t:tokens)logits=full.step(t);}}
    }
    auto end=std::chrono::steady_clock::now();double ms=std::chrono::duration<double,std::milli>(end-prefill).count();
    std::cout<<"{\"text\":\""<<escaped(prompt+generated)<<"\",\"generated_tokens\":"<<steps<<",\"cached\":"<<(cached?"true":"false")<<",\"quantized\":"<<(model.quantized?"true":"false")<<",\"weight_bytes\":"<<model.bytes()<<",\"kv_capacity_bytes\":"<<2*model.context*model.d*4<<",\"prefill_ms\":"<<std::chrono::duration<double,std::milli>(prefill-begin).count()<<",\"decode_ms\":"<<ms<<",\"tokens_per_second\":"<<(ms>0?steps*1000/ms:0)<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"pocket-transformer: "<<e.what()<<"\n";return 1;}}
