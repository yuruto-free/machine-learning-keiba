#!/bin/bash

export USER=labuser
export HOME=/home/${USER}

# Check PUID and PGID of environment variable
if [ -n "${PUID}" ] && [ -n "${PGID}" ]; then

    # Check value of PUID
    if [ ${PUID} -ne 0 ]; then
        uid=$(id -u ${USER})
        gid=$(id -g ${USER})

        # In the case of gid is not equal to PGID
        if [ ${gid} -ne ${PGID} ]; then
            getent group ${PGID} > /dev/null 2>&1 || groupmod -g ${PGID} ${USER}
            chgrp -R ${PGID} ${HOME}
        fi
        # In the case of uid is not equal to PUID
        if [ ${uid} -ne ${PUID} ]; then
            usermod -u ${PUID} ${USER}
        fi
    fi
fi

# Run the command as an USER
exec setpriv --reuid=${USER} --regid=${USER} --init-groups "$@"
