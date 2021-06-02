#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 1 ]; then
  echo "Usage: $0 [--remove-archive] <data-base-folder>"
  echo "e.g.: $0 https://www.openslr.org/resources/17/musan.tar.gz"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<database-name> can be one of: musan.tar.gz, sim_rir_16k.zip"
fi

data=$1
url="https://www.openslr.org/resources/17/musan.tar.gz"

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/musan/.complete ]; then
  echo "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi

pushd $data

if [ ! -f musan.tar.gz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

if ! tar -xvzf musan.tar.gz; then
  echo "$0: error un-tarring archive $data/musan.tar.gz"
  exit 1;
fi

popd >&/dev/null

touch $data/musan/.complete

echo "$0: Successfully downloaded and un-tarred $data/musan.tar.gz"

if $remove_archive; then
  echo "$0: removing $data/musan.tar.gz file since --remove-archive option was supplied."
  rm $data/musan.tar.gz
fi 
