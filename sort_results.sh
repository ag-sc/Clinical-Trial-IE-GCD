configdate=$(date '+%Y-%m-%d')
cd /path/to/project/root
mkdir -p ./results_"$configdate"/gen/basic/dm2-t5
mkdir -p ./results_"$configdate"/gen/basic/dm2-led
mkdir -p ./results_"$configdate"/gen/basic/gl-t5
mkdir -p ./results_"$configdate"/gen/basic/gl-led

mkdir -p ./results_"$configdate"/gen/ptrmodel/dm2-t5
mkdir -p ./results_"$configdate"/gen/ptrmodel/dm2-led
mkdir -p ./results_"$configdate"/gen/ptrmodel/gl-t5
mkdir -p ./results_"$configdate"/gen/ptrmodel/gl-led

mv ./*gen_model_basic_dm2_flan-t5* ./config_gen_dm2_flan-t5*_basic.json* ./results_"$configdate"/gen/basic/dm2-t5/
mv ./*gen_model_basic_dm2_led* ./config_gen_dm2_led*_basic.json* ./results_"$configdate"/gen/basic/dm2-led/
mv ./*gen_model_basic_gl_flan-t5* ./config_gen_gl_flan-t5*_basic.json* ./results_"$configdate"/gen/basic/gl-t5/
mv ./*gen_model_basic_gl_led* ./config_gen_gl_led*_basic.json* ./results_"$configdate"/gen/basic/gl-led/

mv ./*gen_model_ptrmodel_dm2_flan-t5* ./config_gen_dm2_flan-t5*_ptrmodel.json* ./results_"$configdate"/gen/ptrmodel/dm2-t5/
mv ./*gen_model_ptrmodel_dm2_led* ./config_gen_dm2_led*_ptrmodel.json* ./results_"$configdate"/gen/ptrmodel/dm2-led/
mv ./*gen_model_ptrmodel_gl_flan-t5* ./config_gen_gl_flan-t5*_ptrmodel.json* ./results_"$configdate"/gen/ptrmodel/gl-t5/
mv ./*gen_model_ptrmodel_gl_led* ./config_gen_gl_led*_ptrmodel.json* ./results_"$configdate"/gen/ptrmodel/gl-led/