static int samldb_next_step(struct samldb_ctx *ac)
{
	if (ac->curstep->next) {
		ac->curstep = ac->curstep->next;
		return ac->curstep->fn(ac);
	}
	/* We exit the samldb module here. If someone set an "ares" to forward
	 * controls and response back to the caller, use them. */
	if (ac->ares) {
		return ldb_module_done(ac->req, ac->ares->controls,
				       ac->ares->response, LDB_SUCCESS);
	} else {
		return ldb_module_done(ac->req, NULL, NULL, LDB_SUCCESS);
	}
}