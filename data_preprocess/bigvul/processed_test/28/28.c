int do_print(const char *path_p, const struct stat *st, int walk_flags, void *unused)
{
	const char *default_prefix = NULL;
	acl_t acl = NULL, default_acl = NULL;
	int error = 0;
	if (walk_flags & WALK_TREE_FAILED) {
		fprintf(stderr, "%s: %s: %s\n", progname, xquote(path_p, "\n\r"),
			strerror(errno));
		return 1;
	}
	/*
	 * Symlinks can never have ACLs, so when doing a physical walk, we
	 * skip symlinks altogether, and when doing a half-logical walk, we
	 * skip all non-toplevel symlinks. 
	 */
	if ((walk_flags & WALK_TREE_SYMLINK) &&
	    ((walk_flags & WALK_TREE_PHYSICAL) ||
	     !(walk_flags & (WALK_TREE_TOPLEVEL | WALK_TREE_LOGICAL))))
		return 0;
	if (opt_print_acl) {
		acl = acl_get_file(path_p, ACL_TYPE_ACCESS);
		if (acl == NULL && (errno == ENOSYS || errno == ENOTSUP))
			acl = acl_get_file_mode(path_p);
		if (acl == NULL)
			goto fail;
	}
	if (opt_print_default_acl && S_ISDIR(st->st_mode)) {
		default_acl = acl_get_file(path_p, ACL_TYPE_DEFAULT);
		if (default_acl == NULL) {
			if (errno != ENOSYS && errno != ENOTSUP)
				goto fail;
		} else if (acl_entries(default_acl) == 0) {
			acl_free(default_acl);
			default_acl = NULL;
		}
	}
	if (opt_skip_base &&
	    (!acl || acl_equiv_mode(acl, NULL) == 0) && !default_acl)
		return 0;
	if (opt_print_acl && opt_print_default_acl)
		default_prefix = "default:";
	if (opt_strip_leading_slash) {
		if (*path_p == '/') {
			if (!absolute_warning) {
				fprintf(stderr, _("%s: Removing leading "
					"'/' from absolute path names\n"),
				        progname);
				absolute_warning = 1;
			}
			while (*path_p == '/')
				path_p++;
		} else if (*path_p == '.' && *(path_p+1) == '/')
			while (*++path_p == '/')
				/* nothing */ ;
		if (*path_p == '\0')
			path_p = ".";
	}
	if (opt_tabular)  {
		if (do_show(stdout, path_p, st, acl, default_acl) != 0)
			goto fail;
	} else {
		if (opt_comments) {
			printf("# file: %s\n", xquote(path_p, "\n\r"));
			printf("# owner: %s\n",
			       xquote(user_name(st->st_uid, opt_numeric), " \t\n\r"));
			printf("# group: %s\n",
			       xquote(group_name(st->st_gid, opt_numeric), " \t\n\r"));
		}
		if (acl != NULL) {
			char *acl_text = acl_to_any_text(acl, NULL, '\n',
							 print_options);
			if (!acl_text)
				goto fail;
			if (puts(acl_text) < 0) {
				acl_free(acl_text);
				goto fail;
			}
			acl_free(acl_text);
		}
		if (default_acl != NULL) {
			char *acl_text = acl_to_any_text(default_acl, 
							 default_prefix, '\n',
							 print_options);
			if (!acl_text)
				goto fail;
			if (puts(acl_text) < 0) {
				acl_free(acl_text);
				goto fail;
			}
			acl_free(acl_text);
		}
	}
	if (acl || default_acl || opt_comments)
		printf("\n");
cleanup:
	if (acl)
		acl_free(acl);
	if (default_acl)
		acl_free(default_acl);
	return error;
fail:
	fprintf(stderr, "%s: %s: %s\n", progname, xquote(path_p, "\n\r"),
		strerror(errno));
	error = -1;
	goto cleanup;
}