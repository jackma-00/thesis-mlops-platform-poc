application_start:
	cd tracking-server \
	&& gnome-terminal -- sh -c "bash -c \"make; exec bash\"" \
	&& cd .. \
	&& cd workflow \
	&& gnome-terminal -- sh -c "bash -c \"make; exec bash\"" \
	&& gnome-terminal -- sh -c "bash -c \"make trigger; exec bash\"" \
	&& cd .. \
	&& cd inference-env \
	&& gnome-terminal -- sh -c "bash -c \"make; exec bash\""
