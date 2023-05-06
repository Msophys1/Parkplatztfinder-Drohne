# based on https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    # Für die Fotos von Kaggle angepasst
    for xml_file in glob.glob(path + '/*.xml'):
        # Das XML am Schluss entfernen
        filename_w_o_ex = os.path.splitext(xml_file)[0]
        # Den Namen des Filmes + die Endung Jpg
        filename = os.path.basename(filename_w_o_ex) + ".jpg"
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('space'):
            occipied = member.get('occupied')
            if occipied == '0':
                label_class = 'vacant'
                
                # Das Zentrum des Rechtecks und die Grösse ist im XML angeben
                center = member[0][0]
                size = member[0][1]

                cent_X = int(center.attrib['x'])
                cent_Y = int(center.attrib['y'])
                w_rect = int(size.attrib['w'])
                h_rect = int(size.attrib['h'])

                # Aufgrund von Verdrehung die Eckpunkte mithilfe von dem Zentrum und der Grösse berechnen
                xmin = cent_X - w_rect/2
                ymin = cent_Y - h_rect/2
                xmax = cent_X + w_rect/2
                ymax = cent_Y + h_rect/2

                # In der Variabel Value speichern
                value = (filename,
                         1280,
                         720,
                         label_class,
                         xmin,
                         ymin,
                         xmax,
                         ymax
                         )
                xml_list.append(value)    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
