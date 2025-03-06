void acl_mask_perm_str(acl_t acl, char *str)
{
	acl_entry_t entry;
	str[0] = '\0';
	if (acl_get_entry(acl, ACL_FIRST_ENTRY, &entry) != 1)
		return;
	for(;;) {
		acl_tag_t tag;
		acl_get_tag_type(entry, &tag);
		if (tag == ACL_MASK) {
			acl_perm_str(entry, str);
			return;
		}
		if (acl_get_entry(acl, ACL_NEXT_ENTRY, &entry) != 1)
			return;
	}
}