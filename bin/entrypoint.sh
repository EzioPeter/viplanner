figlet VIPlanner

#==
# Log into the container as the host user
#==
set home for host user
export HOME=/home/$HOST_USERNAME
export USER=$HOST_USERNAME
cd $HOME

# Enable sudo access without password
echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
echo "$HOST_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# # Proceed as host user with superuser permissions
sudo -E -u $HOST_USERNAME bash --rcfile /etc/bash.bashrc
