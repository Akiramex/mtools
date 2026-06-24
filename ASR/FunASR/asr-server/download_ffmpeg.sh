wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared.tar.xz
tar -xvf ffmpeg-N-111383-g20b8688092-linux64-gpl-shared.tar.xz

echo 'export PATH=/app/ffmpeg/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared/bin:$PATH' >> /etc/profile
sources /etc/profile