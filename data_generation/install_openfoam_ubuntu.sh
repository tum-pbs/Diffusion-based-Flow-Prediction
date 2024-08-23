# Install gmsh
# Other version of gmsh should also work with the scripts, but not guaranteed.
sudo apt-get install gmsh=4.8.4+ds2-2build1

# support ubuntu 18.04 LTS, 20.04 LTS, 22.04 LTS and 22.10
# see more at https://openfoam.org/version/10/
# Other version of OpenFoam should also work with the scripts, but not guaranteed.
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam10


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Adding environment variable of OpenFOAM to to ~/.bashrc file."
echo "The environment variable will only be available for current terminal."
echo 'As it shown above, please add `source "/opt/openfoam10/etc/bashrc"` to your `~/.bashrc file.`'
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
source "/opt/openfoam10/etc/bashrc"
