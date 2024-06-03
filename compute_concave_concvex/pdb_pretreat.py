import numpy as np
import os
import Bio
import shutil
import collections
from Bio.PDB import *
import sys
import importlib
from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace
from BSSPPIopts import BSSPPIopts
from Bio.PDB import *
from Bio.SeqUtils import IUPACData
PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]
from BSSPPIopts import radii, polarHydrogens
import random
import pymesh
import time
from numpy.linalg import norm
from compute import compute_geo,dictsum,dictchu
# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1"


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set

def protonate(in_pdb_file, out_pdb_file):
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

def extractPDB(
    infilename, outfilename, chain_ids=None
):
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def output_pdb_as_xyzrn(pdbfilename, xyzrnfilename):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if atomtype in radii and resname in polarHydrogens:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")


def read_msms(file_root):
    # read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face

    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    res_id = [""] * count["vertices"]
    res_id_1 = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        res_id[vi] = fields[7]
        res_id_1[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, res_id, res_id_1

def computeMSMS(pdb_file):
    msms_bin = 'msms'
    randnum = random.randint(1,10000000)
    file_base = BSSPPIopts['tmp_dir']+"/msms_"+str(randnum)
    out_xyzrn = file_base+".xyzrn"

    output_pdb_as_xyzrn(pdb_file, out_xyzrn)

    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, names, orinames = read_msms(file_base)

    # Remove temporary files.
    os.remove(file_base+'.area')
    os.remove(file_base+'.xyzrn')
    os.remove(file_base+'.vert')
    os.remove(file_base+'.face')
    return vertices, faces, names, orinames


def fix_mesh(mesh, resolution=1.0, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2

    target_len = resolution
    # print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first
    mesh,_ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    count = 0
    print("Removing degenerated triangles")
    mesh,__ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh,__ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh,1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh,target_len,preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh,150.0,100)
        if mesh.num_vertices == num_vertices:
            break
        num_vertices = mesh.num_vertices
        count+=1
        if count>10: break
    
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh,__ = pymesh.remove_duplicated_faces(mesh)
    mesh,__ = pymesh.remove_obtuse_triangles(mesh,179.0,5)
    mesh,__ = pymesh.remove_isolated_vertices(mesh)
    mesh,_ = pymesh.remove_duplicated_vertices(mesh,0.001)

    return mesh

# Save the chains as separate files.
time_start = time.time()
in_field = sys.argv[1]
print(in_field)
if "_" not in in_field:
	in_fields = in_field.strip()
	pdb_id = in_fields
	chain_ids1 = "A"
else:
	in_fields = in_field.strip().split('_')
	pdb_id = in_fields[0]
	chain_ids1 = in_fields[1]
	
pdb_filename = "./data_preparation/raw_pdbs/"+pdb_id+".pdb"
tmp_dir= BSSPPIopts['tmp_dir']
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
'''
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file
'''
print(pdb_filename)

# Extract chains of interest.

out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
outfilename = pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Compute MSMS of surface
vertices1, faces1, names1, orinames= computeMSMS(out_filename1+".pdb")
vert = {}
for i in range(len(vertices1)):
    items = {str(vertices1[i][0])+str(vertices1[i][1])+str(vertices1[i][2]):names1[i]}
    vert.update(items)

original_mesh = pymesh.form_mesh(vertices1, faces1)
mesh = original_mesh
#mesh = fix_mesh(original_mesh)
mesh,_ = pymesh.remove_duplicated_vertices(mesh,0.001)
mesh,__ = pymesh.remove_degenerated_triangles(mesh,100)
mesh.add_attribute("vertex_mean_curvature")
mesh.add_attribute("vertex_gaussian_curvature")
mesh.add_attribute("vertex_normal")
mesh.add_attribute("face_centroid")
mesh.add_attribute("face_circumcenter")
mesh.add_attribute("face_normal")
vertices2 = mesh.vertices
newvert={}
for i in vertices2:
    items = {str(i[0])+str(i[1])+str(i[2]):1}
    newvert.update(items)
newname = []
for key in newvert.keys():
    newname.append(vert.get(key))
mesh.add_attribute("vertex_name")
mesh.set_attribute("vertex_name",np.array(newname))
pymesh.save_mesh(out_filename1+".ply",mesh,*mesh.get_attribute_names(),use_float=True,ascii=True)

namedict = {}
gauscurva = mesh.get_attribute("vertex_gaussian_curvature")
meancurva = mesh.get_attribute("vertex_mean_curvature")
for i in range(len(names1)):
    items = {float(names1[i]):orinames[i]}
    namedict.update(items)

curdict = {}
for i in range(len(newname)):
    tmpname = float(newname[i])
    if tmpname in curdict:
        curdict[tmpname] = dictsum(curdict[tmpname],[gauscurva[i],meancurva[i]])
    else:
        curdict.update({tmpname:[gauscurva[i],meancurva[i],1]})

for key in curdict.keys():
    curdict[key] = dictchu(curdict[key])

vertices = mesh.vertices
resdiue_id = mesh.get_attribute("vertex_name")
norm = mesh.get_attribute("vertex_normal")
vert = {}
for i in range(len(vertices)):
    items = {i:str(norm[i*3+0])+" "+str(norm[i*3+1])+" "+str(norm[i*3+2])+"\t"+str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\t"+str(resdiue_id[i])}
    vert.update(items)

faces = mesh.faces
centroid = mesh.get_attribute("face_centroid")
normal = mesh.get_attribute("face_normal")
circumcenter = mesh.get_attribute("face_circumcenter")

for i in range(len(faces)):
    for subface in faces[i]:
        vert[subface] = vert[subface]+"\t"+ str(normal[i*3+0])+" "+str(normal[i*3+1])+" "+str(normal[i*3+2])+" "+str(centroid[i*3+0])+" "+str(centroid[i*3+1])+" "+str(centroid[i*3+2])+" "+str(circumcenter[i*3+0])+" "+str(circumcenter[i*3+1])+" "+str(circumcenter[i*3+2])

geo, every_center = compute_geo(vert)
time_end = time.time()
print('totally cost',time_end-time_start)
print('time per molecule',(time_end-time_start)/len(geo))
for key in geo.keys():
    listgeo = str(geo[key]).split('[')[1].split(']')[0].split(',')
    listgeo.append(curdict[key][0])
    listgeo.append(curdict[key][1])
    listgeo.append(namedict[key])
    geo[key] = listgeo

codes = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}


if not os.path.exists("./oridata"):
    os.makedirs("./oridata")

fp1 = open("./oridata/"+outfilename+".data","w+",encoding="utf-8")

geometry_sum = sorted(geo.items(),key = lambda d:d[0])
for i in range(len(geometry_sum)):
    for j in geometry_sum[i][1]:
        if "_" in str(j):
            fp1.write(str(j).split("_")[1]+"_"+str(codes[str(j).split("_")[3]])+"\t"+str(every_center[geometry_sum[i][0]])+"\n")
        else: 
            fp1.write(str(j)+"\t")


