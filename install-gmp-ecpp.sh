#!/bin/bash

here="`pwd`"
topdir="`dirname ${here}`"
destdir="${topdir}/install-gmp-ecpp"
rc=0

if [ ! -d ${destdir} ] ; then
  mkdir -p ${destdir}
fi

prefix="/usr/local"
install_bindir="${prefix}/bin"
install_libdir="${prefix}/lib64"
install_incdir="${prefix}/include"

mkdir -p ${destdir}/${install_bindir}
mkdir -p ${destdir}/${install_libdir}
mkdir -p ${destdir}/${install_incdir}

cuda_version="13.2"
cuda_libdir="/usr/local/cuda-${cuda_version}/lib64"
runpath="${install_libdir}:${cuda_libdir}"

for file in \
  gmp-ecpp \
  gmpecpp-bench \
  ecm-reach \
  kernel-latency \
  gmpecpp-verify
do
  if [ -f ${here}/${file} ] ; then
    echo "patchelf --set-rpath ${runpath} ${here}/${file}"
    patchelf --set-rpath ${runpath} ${here}/${file}
    echo "cp -f ${here}/${file} ${destdir}/${install_bindir}/"
    cp -f ${here}/${file} ${destdir}/${install_bindir}/
  else
    echo "expected program ${file} was not found!"
    exit 1
  fi
done

tarball="libgmpecpp.so.1.0.0.tar"
xztarball="${tarball}.xz"

if [ ! -f ${here}/libgmpecpp.so.1.0.0 ] ; then
  echo "shared library libgmpecpp.so.1.0.0 file was not found!"
  exit 1
fi

if [ ! -L ${here}/libgmpecpp.so.1 ] ; then
  echo "shared library symlink libgmpecpp.so.1 was not found!"
  exit 1
fi

if [ ! -L ${here}/libgmpecpp.so ] ; then
  echo "shared library symlink libgmpecpp.so was not found!"
  exit 1
fi

if [ -e ${here}/${tarball} ] ; then
  rm -f ${here}/${tarball}
fi

if [ -e ${here}/${xztarball} ] ; then
  rm -f ${here}/${xztarball}
fi

echo "patchelf --set-rpath ${runpath} libgmpecpp.so.1.0.0"
patchelf --set-rpath ${runpath} libgmpecpp.so.1.0.0

tar cf ${tarball} libgmpecpp.so.1.0.0 libgmpecpp.so.1 libgmpecpp.so

if [ ! -f ${here}/${tarball} ] ; then
  echo "shared library tarball ${tarball} was not created!"
  exit 1
fi

xz -z -T 8 ${tarball}

if [ ! -f ${here}/${xztarball} ] ; then
  echo "compressed shared library tarball ${xztarball} was not created!"
  exit 1
fi

echo "cp -fp ${xztarball} ${destdir}/${install_libdir}"
cp -fp ${xztarball} ${destdir}/${install_libdir}

echo "cd ${destdir}/${install_libdir}"
cd ${destdir}/${install_libdir}

echo "tar xf ${xztarball}"
tar xf ${xztarball}
rc=$?

if [ ${rc} -ne 0 ] ; then
  echo "Uncompressing the XZ Tarball failed!"
  exit 1
fi

echo "rm -f ${xztarball}"
rm -f ${xztarball}

cd ${here}

echo "cp -fp ${here}/include/gmpecpp.h ${destdir}/${install_incdir}/"
cp -fp ${here}/include/gmpecpp.h ${destdir}/${install_incdir}/

echo "Installation Done."




