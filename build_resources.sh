wget https://github.com/xg-chu/lightning_track/releases/download/resources/resources.tar ./resources.tar
tar -xvf resources.tar
mv resources/emoca/* .flame_feature_extractor/feature_extractor/emoca/assets/
mv resources/FLAME/* ./flame_feature_extractor/renderer/assets/
#mv resources/human_matting/* ./engines/human_matting/assets/
mv resources/mica/* ./flame_feature_extractor/feature_extractor/mica/assets/
#rm -r resources/