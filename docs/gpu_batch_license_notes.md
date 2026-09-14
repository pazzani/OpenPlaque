# GPU batch TotalSegmentator license handling

The unattended GPU batch reads the TotalSegmentator license from `/content/drive/MyDrive/OpenPlaque/private/totalseg_license.txt` after Google Drive is mounted, activates TotalSegmentator at runtime with `totalseg_set_license`, and never commits or prints the license value. The batch runner receives only an `ACTIVE` sentinel after activation so licensed tasks can be enabled without exposing the license in GitHub, reports, or command logs.
